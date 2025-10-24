import asyncio
import os
import random
import traceback
from datetime import datetime
from time import time
from urllib.parse import quote

import httpx
import numpy as np
import openai
import pandas as pd
# Socket.IO & Motor (async Mongo)
import socketio
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.responses import JSONResponse

# --- Load Environment ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
openai.api_key = OPENAI_API_KEY

WHATSAPP_ACCESS_TOKEN = "EAAQYybD4vYYBPkO0ZCsdI7j4i8qPwkc0p6yyqcJILxYQkrj3MB9PuHXHa7ZAUCaWJLocryOdfHKkCV23LZAWxlMWFxzu9mUCQQhAJugijcTFkcNMzfCaPbT6hDqyaW4xkjinvtMxziTVhMiOXBOVMHe64OrRQEahCZBzENChcAeZAdug9sZBaHltSAYbSbDNdkQgZDZD"
PHONE_NUMBER_ID = "836070512923409"

# --- Google Sheets (Excel-style) File ---
SHEET_ID = "1FO8Gb703ipZrWrSxTe2XTsBnv345_jwtJHBycLA-4Vo"
SHEET_NAMES = ["handshaking", "golden visa"]

ADMIN_NUMBERS = ["306980102740", "923244181389"]

bot_active = True
global_threshold = {"value": 0.29}
global_rag = {"value": "gpt-4o-mini"}
global_top_k = {"value": 5}
global_temperature = {"value": 0.2}
model_name = "text-embedding-3-small"

# ----------------------------
# --- MongoDB (motor async)
# ----------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DBNAME = os.getenv("MONGO_DBNAME", "whatsapp_chat")

mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client[MONGO_DBNAME]
messages_collection = db["messages"]
configs_collection = db['configs']

# ----------------------------
# --- Socket.IO server
# ----------------------------
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
# We'll wrap FastAPI with socketio ASGI app later

# --- Load Dataset from Google Sheets ---
def load_dataset_from_google_sheet(sheet_id):
    all_data = []
    global system_prompt_text

    for sheet_name in SHEET_NAMES:
        safe_name = quote(sheet_name)
        print(f"📄 Loading sheet: {sheet_name}")
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={safe_name}"
        df = pd.read_csv(url)
        df["source_sheet"] = sheet_name
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ Loaded {len(combined_df)} rows from {len(SHEET_NAMES)} sheets.")

    # Load system prompt (only cell A1)
    print("📄 Loading sheet: prompt")
    system_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet=prompt"
    prompt_df = pd.read_csv(system_url, header=None, dtype=str)

    if not prompt_df.empty:
        system_prompt_text = str(prompt_df.iloc[0, 0]).strip()
        print("🧠 Loaded system prompt from Google Sheet:")
        print(system_prompt_text)
    else:
        system_prompt_text = (
            "Sen profesyonel ama samimi bir emlak danışmanısın. "
            "Doğal, içten ve insana benzeyen bir dil kullan; yapay veya ezberlenmiş gibi konuşma. "
            "Cevaplarını kısa, açık ve dostça tut — tıpkı bir insanla konuşuyormuşsun gibi. "
            "Yanıtlarını yalnızca verilen 'Bağlam' (context) içindeki bilgilere dayanarak oluştur. "
            "Bağlamda ilgili bilgi varsa, onu doğal şekilde kullanarak cevap ver. "
            "Bağlamda tam bir yanıt yoksa, genel bir ifade ile yardımcı olmaya çalış ama tahmin yürütme veya yeni bilgi uydurma. "
            "Eğer gerçekten emin değilsen, 'Bundan emin değilim.' diyebilirsin. "
            "Kendi bilgi bankanı veya dış kaynakları kullanma — sadece verilen bağlama güven. "
            "Kullanıcının sorduğu dili algıla ve cevabı aynı dilde ver (örnek: soru İngilizce ise yanıt da İngilizce olmalı)."
        )
        print("⚠️ Warning: 'prompt' sheet is empty. Using default prompt.")

    return combined_df


# --- Build Embedding Index (OpenAI) ---
def build_index(df, model_name="text-embedding-3-small"):
    print("📦 Encoding dataset with OpenAI embeddings...")
    start = time()

    texts = df["questions"].astype(str).tolist()
    all_embeddings = []

    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = openai.embeddings.create(
            input=batch,
            model=model_name
        )
        batch_embeddings = [d.embedding for d in response.data]
        all_embeddings.extend(batch_embeddings)

    embeddings = np.array(all_embeddings, dtype="float32")
    elapsed = time() - start
    print(f"✅ Encoded {len(texts)} entries in {elapsed:.2f} seconds")
    return embeddings, texts


# --- Semantic Search (Weighted Multi-Result) ---
def semantic_search(user_query, df, embeddings, texts, model_name="text-embedding-3-small", top_k=5, threshold=0.5):
    print(f"\n🔎 Semantic search started for query: '{user_query}' with model: '{model_name}'")

    # --- Get query embedding ---
    try:
        response = openai.embeddings.create(
            input=[user_query],
            model=model_name
        )
        # Handle both large and small model responses safely
        if hasattr(response, "data") and len(response.data) > 0:
            query_vec = response.data[0].embedding
        elif isinstance(response, dict) and "data" in response and len(response["data"]) > 0:
            query_vec = response["data"][0]["embedding"]
        else:
            raise ValueError("Embedding response structure unexpected.")
    except Exception as e:
        print(f"⚠️ Error creating embedding: {e}")
        traceback.print_exc()
        return None, []

    query_vec = np.array(query_vec, dtype="float32")

    # --- Compute cosine similarities ---
    sim_scores = np.dot(embeddings, query_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-10
    )

    # --- Select top-k results ---
    top_idx = np.argsort(sim_scores)[::-1][:top_k]
    results = []

    print("📊 Top similarity results:")
    for i in top_idx:
        q = str(df.iloc[i]["questions"])
        a = str(df.iloc[i]["answers"])
        s = float(sim_scores[i])
        print(f"  → Q: {q[:80]}... | Score={s:.4f}")
        results.append((q, a, s))

    best_score = float(np.max(sim_scores)) if len(sim_scores) > 0 else 0.0
    print(f"🏁 Best score: {best_score:.4f} | Threshold: {threshold}")

    # --- Handle threshold ---
    if best_score < threshold:
        print("⚠️ No match exceeded the threshold.\n")
        return None, results

    # --- Combine top answers if multiple are relevant ---
    top_answers = [a for _, a, s in results if s >= threshold]
    if len(top_answers) == 0:
        top_answers = [results[0][1]]  # fallback: best answer only

    combined_answer = " ".join(top_answers)
    print(f"✅ Match found above threshold. Using average of {len(top_answers)} answers.\n")

    return combined_answer, results



# --- RAG Response Generation (Weighted Answers) ---
def generate_rag_response(user_query, results, chat_history):
    if not results:
        return "Bu konuda elimde net bir bilgi bulunmuyor.", "⚠️ No results found."

    # 🎯 Weight answers by similarity score
    total_score = sum(s for _, _, s in results)
    weighted_context_parts = []
    for q, a, s in results:
        weight = s / total_score if total_score > 0 else 0
        weighted_context_parts.append(f"(Ağırlık {weight:.2f}) Soru: {q}\nCevap: {a}")
    context = "\n\n".join(weighted_context_parts)

    # 🕰 Include limited chat history (last 3 exchanges)
    history_str = (
        "\n".join([f"Kullanıcı: {u}\nAsistan: {a}" for u, a, _ in chat_history[-3:]])
        if chat_history else ""
    )

    system_prompt = f"""{system_prompt_text}
    Eğer birden fazla bağlam alakalı görünüyorsa, bunlardan uygun olan bilgileri birleştirerek doğal, tutarlı ve insana benzer bir cevap oluştur.
    [IMPORTANT NOTE] Answer in the same language as user question.
    """
    user_prompt = (
        f"Geçmiş konuşma:\n{history_str}\n\n"
        f"Kullanıcının yeni sorusu: {user_query}\n\n"
        f"Bağlam (dataset'ten alınan bilgiler):\n{context}"
    )

    # 🧠 Debug printout of the exact data sent to OpenAI
    print("\n====================== 🧠 MODEL INPUT DEBUG ======================")
    print("🧩 SYSTEM PROMPT:\n", system_prompt)
    print("\n💬 USER PROMPT:\n", user_prompt)
    print("=================================================================\n")

    try:
        response = openai.chat.completions.create(
            model=global_rag["value"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=global_temperature["value"],
        )

        answer = response.choices[0].message.content.strip()

        # 🪄 Debug printout of model response
        print("\n====================== 🤖 MODEL RESPONSE ======================")
        print(answer)
        print("================================================================\n")

        return answer, context

    except Exception as e:
        print(f"⚠️ Error during model completion: {str(e)}")
        traceback.print_exc()
        return f"⚠️ Hata: {str(e)}", context



# --- Initial Load ---
df = load_dataset_from_google_sheet(SHEET_ID)
embeddings, texts = build_index(df)
chat_sessions = {}

# ----------------------------
# --- FastAPI + CORS
# ----------------------------
fastapi_app = FastAPI()
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# --- integrate socketio with fastapi
# ----------------------------
# socketio ASGI app wraps the FastAPI app at the end of the file
# use global `sio` to emit events

# ----------------------------
# --- Helpers for message storage & emit
# ----------------------------
async def emit_new_message(doc):
    """Broadcast a new message to any connected dashboard clients"""
    payload = {
        "clientNumber": doc["clientNumber"],
        "message": doc["message"],
        "direction": doc["direction"],
        "outgoingSender": doc.get("outgoingSender"),
        "timestamp": doc["timestamp"].isoformat() if isinstance(doc["timestamp"], datetime) else str(doc["timestamp"])
    }
    # emit to channel 'new_message'
    await sio.emit("new_message", payload)
    print("🔊 Emitted new_message:", payload)


async def save_message_and_emit(client_number, direction, message, outgoing_sender=None):
    """Save message in MongoDB and notify dashboard via socketio"""
    doc = {
        "clientNumber": client_number,
        "message": message,
        "direction": direction,
        "timestamp": datetime.utcnow()
    }
    if direction == "outgoing" and outgoing_sender:
        doc["outgoingSender"] = outgoing_sender
    res = await messages_collection.insert_one(doc)
    doc["_id"] = str(res.inserted_id)
    await emit_new_message(doc)
    print("💾 Saved message:", doc)
    return doc

# ----------------------------
# --- Your existing webhook endpoints (modified)
# ----------------------------
@fastapi_app.get("/webhook")
async def verify(request: Request):
    params = request.query_params
    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == VERIFY_TOKEN
    ):
        return int(params.get("hub.challenge"))
    return "Verification failed", 403


@fastapi_app.post("/webhook")
async def receive(request: Request):
    global df, embeddings, texts
    global bot_active
    global model_name

    data = await request.json()
    print("📩 Incoming message:", data)

    try:
        entry = data["entry"][0]["changes"][0]["value"]
        messages = entry.get("messages")
        if not messages:
            return "EVENT_RECEIVED", 200

        msg = messages[0]
        from_number = msg["from"]
        # Don't lower-case message for storage — keep original for display. Keep lowercase for semantic search if desired.
        raw_text = msg.get("text", {}).get("body", "")
        text_for_search = raw_text.strip().lower()
        # Save incoming client message
        await save_message_and_emit(from_number, "incoming", raw_text)

        # Check if bot is enabled for this client
        config = await configs_collection.find_one({"clientNumber": from_number})
        if config and not config.get("botEnabled", True):
            print(f"🤖 Bot disabled for {from_number} — ignoring message.")
            return "BOT_DISABLED_FOR_CLIENT", 200

        # --- Admin commands (from admins via WhatsApp) ---
        if from_number in ADMIN_NUMBERS:
            if text_for_search.startswith("threshold="):
                try:
                    new_threshold = float(text_for_search.split("=", 1)[1])
                    global_threshold["value"] = new_threshold
                    await send_whatsapp_message(from_number, f"✅ Threshold updated to {new_threshold}")
                    print(f"⚙️ Threshold updated to {new_threshold} by admin.")
                except ValueError:
                    await send_whatsapp_message(from_number, "⚠️ Invalid threshold format. Use like: threshold=0.35")
                return "THRESHOLD_UPDATED", 200

            # Top_k update
            if text_for_search.startswith("top_k="):
                try:
                    new_top_k = int(text_for_search.split("=", 1)[1])
                    global_top_k["value"] = new_top_k
                    await send_whatsapp_message(from_number, f"✅ top_k updated to {new_top_k}")
                    print(f"⚙️ top_k updated to {new_top_k} by admin.")
                except ValueError:
                    await send_whatsapp_message(from_number, "⚠️ Invalid top_k format. Use like: top_k=2")
                return "TOPK_UPDATED", 200

            if text_for_search.startswith("temperature="):
                try:
                    new_temperature = float(text_for_search.split("=", 1)[1])
                    global_temperature["value"] = new_temperature
                    await send_whatsapp_message(from_number, f"✅ temperature updated to {new_temperature}")
                    print(f"⚙️ temperature updated to {new_temperature} by admin.")
                except ValueError:
                    await send_whatsapp_message(from_number, "⚠️ Invalid temperature format. Use like: temperature=0.1")
                return "TEMPERATURE_UPDATED", 200

            if text_for_search == "status":
                status_message = (
                    f"📊 *Current Bot Configuration:*\n"
                    f"• Threshold: {global_threshold['value']}\n"
                    f"• Top K: {global_top_k['value']}\n"
                    f"• Temperature: {global_temperature['value']}\n"
                    f"• Bot Active: {'✅ Yes' if bot_active else '⏸ No'}"
                )
                await send_whatsapp_message(from_number, status_message)
                print(f"ℹ️ Status requested by admin: {status_message}")
                return "STATUS_SENT", 200

            if text_for_search.startswith("rag="):
                try:
                    new_rag_model = text_for_search.split("=", 1)[1].strip()
                    global_rag["value"] = new_rag_model  # store as string
                    await send_whatsapp_message(from_number, f"✅ Rag model updated to {new_rag_model}")
                    print(f"⚙️ Rag model updated to {new_rag_model} by admin.")
                except Exception as e:
                    await send_whatsapp_message(from_number, f"⚠️ Failed to update RAG model: {e}")
                return "THRESHOLD_UPDATED", 200

            if text_for_search == "prompt":
                prompt_message = f"📊 *Current RAG Prompt:*\n{system_prompt_text}"
                await send_whatsapp_message(from_number, prompt_message)
                print(f"ℹ️ prompt requested by admin: {prompt_message}")
                return "PROMPT_SENT", 200

            if text_for_search == "stop":
                bot_active = False
                await send_whatsapp_message(from_number, "⏸ Bot paused globally.")
                print("🚫 Bot paused by admin.")
                return "BOT_PAUSED", 200

            if text_for_search == "start":
                bot_active = True
                await send_whatsapp_message(from_number, "▶️ Bot resumed globally.")
                print("✅ Bot resumed by admin.")
                return "BOT_RESUMED", 200

            if text_for_search.startswith("refresh"):
                print("🔄 Admin requested dataset refresh...")
                df = load_dataset_from_google_sheet(SHEET_ID)

                if "2" in text_for_search:
                    model_name = "text-embedding-3-large"
                else:
                    model_name = "text-embedding-3-small"

                embeddings, texts = build_index(df, model_name=model_name)
                await send_whatsapp_message(from_number, "🔁 Dataset refreshed successfully.")
                print("✅ Dataset refreshed.")
                return "REFRESH_OK", 200

        # --- If bot is paused globally ---
        if not bot_active:
            print("🤖 Bot is paused — ignoring messages.")
            return "BOT_INACTIVE", 200

        # --- Regular user flow ---
        delay = random.uniform(5, 10)
        print(f"⏳ Simulating human typing... waiting {delay:.2f} seconds")
        await asyncio.sleep(delay)

        chat_history = chat_sessions.get(from_number, [])
        combined_answer, results = semantic_search(
            text_for_search, df, embeddings, texts,
            model_name=model_name,
            top_k=global_top_k["value"],
            threshold=global_threshold["value"]
        )

        if combined_answer is None:
            print("🔍 No strong match found.")
            rag_response = (
                "Bu konuda elimde net bir bilgi bulunmuyor. "
                "Ben yalnızca **Yunanistan Golden Visa** programı ile ilgili sorulara yardımcı olabiliyorum. 🇬🇷 "
                "Lütfen sorunuz bu konuyla ilgiliyse tekrar yazın. 😊"
            )
            context_used = "⚠️ Veri kümesinde ilgili içerik bulunamadı."
        else:
            rag_response, context_used = generate_rag_response(
                text_for_search, results, chat_history
            )

        chat_history.append((raw_text, rag_response, context_used))
        chat_sessions[from_number] = chat_history

        # --- Send reply to WhatsApp ---
        url = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"
        headers = {
            "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": from_number,
            "text": {"body": rag_response},
        }
        print("📩 Sending to WhatsApp:", payload)
        # Use httpx async to send
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload)
            print("📤 WhatsApp response:", resp.status_code, resp.text)

        # Save outgoing bot message and emit to dashboards
        await save_message_and_emit(from_number, "outgoing", rag_response, outgoing_sender="bot")

    except Exception as e:
        print("⚠️ Error handling message:", e)
        traceback.print_exc()

    return "EVENT_RECEIVED", 200

# create indexes (safe to call repeatedly)
@fastapi_app.on_event("startup")
async def ensure_indexes():
    await messages_collection.create_index([("clientNumber", 1), ("timestamp", 1)])

# --- Admin HTTP endpoint to send message from dashboard ---
@fastapi_app.post("/send")
async def send_from_dashboard(request: Request):
    """
    Body: {"client_number": "923...", "message": "text", "admin_number": "306..."}
    admin_number optional, but validate admin in production.
    """
    body = await request.json()
    client_number = body.get("client_number")
    message = body.get("message")
    admin_number = body.get("admin_number")

    if not client_number or not message:
        return {"error": "client_number and message required"}, 400

    # Optional: validate admin_number in production
    if admin_number and admin_number not in ADMIN_NUMBERS:
        return {"error": "unauthorized admin"}, 403

    # send via existing helper
    await send_whatsapp_message(client_number, message)

    # store as outgoing/admin (no sentBy field in your final schema)
    await save_message_and_emit(client_number, "outgoing", message, outgoing_sender="admin")
    return {"status": "sent"}


# --- Endpoint to list unique conversations (latest message per client) ---
@fastapi_app.get("/conversations")
async def get_conversations():
    pipeline = [
        {"$sort": {"timestamp": -1}},
        {"$group": {
            "_id": "$clientNumber",
            "lastMessage": {"$first": "$message"},
            "direction": {"$first": "$direction"},
            "outgoingSender": {"$first": "$outgoingSender"},
            "lastTimestamp": {"$first": "$timestamp"}
        }},
        {"$sort": {"lastTimestamp": -1}}
    ]
    docs = messages_collection.aggregate(pipeline)
    results = []
    async for doc in docs:
        results.append({
            "clientNumber": doc["_id"],
            "lastMessage": doc.get("lastMessage"),
            "direction": doc.get("direction"),
            "outgoingSender": doc.get("outgoingSender"),
            "lastTimestamp": doc.get("lastTimestamp").isoformat() if doc.get("lastTimestamp") else None
        })
    return results


# --- Get chat history for a client ---
@fastapi_app.get("/chats/{client_number}")
async def get_chat(client_number: str):
    cursor = messages_collection.find({"clientNumber": client_number}).sort("timestamp", 1)
    out = []
    async for doc in cursor:
        out.append({
            "clientNumber": doc["clientNumber"],
            "message": doc["message"],
            "direction": doc["direction"],
            "outgoingSender": doc.get("outgoingSender"),
            "timestamp": doc["timestamp"].isoformat() if isinstance(doc["timestamp"], datetime) else str(doc["timestamp"])
        })
    return {"messages": out}


@fastapi_app.post("/client-bot-toggle")
async def toggle_bot(request: Request):
    data = await request.json()
    client_number = data.get("clientNumber")
    bot_enabled = data.get("botEnabled", True)

    if not client_number:
        return {"error": "clientNumber is required"}

    await configs_collection.update_one(
        {"clientNumber": client_number},
        {"$set": {"botEnabled": bot_enabled}},
        upsert=True
    )
    return {"status": "ok", "botEnabled": bot_enabled}

@fastapi_app.get("/get-client-config")
async def get_client_config(clientNumber: str):
    if not clientNumber:
        return JSONResponse(status_code=400, content={"error": "clientNumber is required"})

    config = await configs_collection.find_one({"clientNumber": clientNumber})
    if not config:
        return {"clientNumber": clientNumber, "botEnabled": True}  # default True

    return {"clientNumber": clientNumber, "botEnabled": config.get("botEnabled", True)}


# --- Helper to send whatsapp messages (async) ---
async def send_whatsapp_message(to, message):
    url = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "text": {"body": message},
    }

    print("📤 Sending message to WhatsApp:", payload)
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=payload)
        print("📤 send_whatsapp_message response:", resp.status_code, resp.text)
        return resp


# --- Socket.IO event handlers (optional) ---
@sio.event
async def connect(sid, environ):
    print("🟢 Socket.IO client connected:", sid)
    await sio.emit("info", {"msg": "connected"}, to=sid)

@sio.event
async def disconnect(sid):
    print("🔴 Socket.IO client disconnected:", sid)


# Wrap FastAPI app with Socket.IO ASGI app
asgi_app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)
app = asgi_app

if __name__ == "__main__":
    print("🚀 Starting server (ASGI app with Socket.IO and FastAPI)")
    uvicorn.run(asgi_app, host="0.0.0.0", port=8000)