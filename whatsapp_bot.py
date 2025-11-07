import asyncio
import imghdr
import csv
import mimetypes
import os
import random
import smtplib
import traceback
from datetime import datetime
from email.message import EmailMessage
from time import time
from urllib.parse import quote
import re
from PIL import Image
import io


import google.generativeai as genai
import httpx
import numpy as np
import openai
import pandas as pd
# Socket.IO & Motor (async Mongo)
import socketio
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Body
from fastapi import Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient

fastapi_app = FastAPI()

# --- Load Environment ---
load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

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
global_rag = {"value": "gpt-4o"}
global_top_k = {"value": 5}
global_temperature = {"value": 0.2}
model_name = "text-embedding-3-small"
processed_message_ids = set()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gen_model = genai.GenerativeModel("gemini-2.5-flash")

PROP360_URL = "https://prop360.pro/api/merchant/form_data"
PROP_AUTH_TOKEN = os.getenv("PROP360_BEARER_TOKEN")

# ----------------------------
# --- MongoDB (motor async)
# ----------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
print(MONGO_URI)
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
    top_results = [(a, s) for _, a, s in results if s >= threshold]

    # --- If no answer passes threshold, return None ---
    if not top_results:
        print("⚠️ No relevant match above threshold — returning None.\n")
        return None, results

    # --- Weight the answers by similarity ---
    scores = np.array([s for _, s in top_results])
    weights = scores / np.sum(scores)  # normalize to sum = 1

    # Combine weighted answers
    weighted_parts = []
    for (answer, score), weight in zip(top_results, weights):
        weighted_parts.append(f"[Weight {weight:.2f}] {answer.strip()}")

    combined_answer = "\n".join(weighted_parts)

    print(f"✅ Weighted combination of {len(top_results)} answers (sum of weights = 1.0)\n")
    return combined_answer, results



# --- RAG Response Generation (Uses Pre-Weighted Combined Answer) ---
def generate_rag_response(user_query, combined_answer, chat_history):
    # 🕰 Include limited chat history (last 3 exchanges)
    history_str = (
        "\n".join([f"User: {u}\nAssistant: {a}" for u, a, _ in chat_history[-3:]])
        if chat_history else ""
    )

    # ✅ Always provide a context, even if no match found
    if not combined_answer:
        combined_answer = (
            "⚠️ No relevant context found in the dataset for this user query. "
            "However, respond naturally but based only on the conversation history."
        )

    system_prompt = f"""{system_prompt_text}
    If multiple contexts seem relevant, combine them appropriately to produce a natural, coherent, human-like answer.
    Each context includes a weight value (e.g., 'Weight 0.87') — treat higher-weighted contexts as more important and prioritize them when forming the response.
    """

    user_prompt = f"""
    🧠 CONVERSATION HISTORY:
    {history_str}

    💬 USER'S NEW QUESTION:
    {user_query}

    📚 CONTEXT (weighted combination or note if none found):
    {combined_answer}

    ⚠️ INSTRUCTIONS:
    - The answer must be entirely in the same language as the user's message.
    - If no relevant context exists, still reply meaningfully but based solely on prior conversations.
    """

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

        return answer, combined_answer

    except Exception as e:
        print(f"⚠️ Error during model completion: {str(e)}")
        traceback.print_exc()
        return f"⚠️ Error: {str(e)}", combined_answer



def generate_text_with_model(input_text, model_name=None, temperature=0.5):
    """
    Sends a plain text prompt to the configured OpenAI model and returns the output text.

    Args:
        input_text (str): The text prompt or question to send to the model.
        model_name (str): Optional. Model to use (default: global_rag["value"] if defined).
        temperature (float): Optional. Sampling temperature for creativity (0.0–1.0).

    Returns:
        str: Model-generated response text, or an error message.
    """

    # Choose default model if not passed
    try:
        model_to_use = model_name or global_rag["value"]
    except NameError:
        model_to_use = "gpt-4o-mini"  # safe fallback

    print("\n====================== 🧠 MODEL INPUT DEBUG ======================")
    print(input_text)
    print("=================================================================\n")

    try:
        response = openai.chat.completions.create(
            model=model_to_use,
            messages=[{"role": "user", "content": input_text}],
            temperature=temperature,
        )

        output_text = response.choices[0].message.content.strip()

        print("\n====================== 🤖 MODEL OUTPUT ======================")
        print(output_text)
        print("================================================================\n")

        return output_text

    except Exception as e:
        print(f"⚠️ Error during model call: {str(e)}")
        traceback.print_exc()
        return f"⚠️ Error: {str(e)}"


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

    if "context" in doc:
        payload["context"] = doc["context"]

    # emit to channel 'new_message'
    await sio.emit("new_message", payload)
    print("🔊 Emitted new_message:", payload)


async def save_message_and_emit(client_number, direction, message, outgoing_sender=None, context=None):
    """Save message in MongoDB and notify dashboard via socketio"""
    doc = {
        "clientNumber": client_number,
        "message": message,
        "direction": direction,
        "timestamp": datetime.utcnow()
    }
    if direction == "outgoing" and outgoing_sender:
        doc["outgoingSender"] = outgoing_sender

    if context is not None:
        doc["context"] = context

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

        message_id = msg["id"]

        # --- Prevent duplicate replies ---
        if message_id in processed_message_ids:
            print(f"⚠️ Duplicate message detected: {message_id} — skipping")
            return "EVENT_RECEIVED", 200
        processed_message_ids.add(message_id)

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

        rag_response, context_used = generate_rag_response(
            text_for_search, combined_answer, chat_history
        )

        detected_language = generate_text_with_model(f"""
        What language is this text in (only return the language name): {raw_text}
        """)

        print("Detected Language:", detected_language)

        translated_text = generate_text_with_model(f"""
        You are a translation assistant.
        If the following text is already written in {detected_language}, return it exactly as is.
        Otherwise, translate it into {detected_language}.
        Return only the translated text — no explanations, no prefixes.

        Text:
        {rag_response}
        """)

        chat_history.append((raw_text, translated_text, context_used))
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
            "text": {"body": translated_text},
        }
        print("📩 Sending to WhatsApp:", payload)
        # Use httpx async to send
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload)
            print("📤 WhatsApp response:", resp.status_code, resp.text)

        # Save outgoing bot message and emit to dashboards
        await save_message_and_emit(from_number, "outgoing", translated_text, outgoing_sender="bot", context=context_used)

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
        client_number = doc["_id"]

        # Fetch name from configs_collection (if it exists)
        client_doc = await configs_collection.find_one({"clientNumber": client_number})
        client_name = client_doc.get("name") if client_doc else ""

        results.append({
            "clientNumber": client_number,
            "clientName": client_name,
            "lastMessage": doc.get("lastMessage"),
            "direction": doc.get("direction"),
            "outgoingSender": doc.get("outgoingSender"),
            "lastTimestamp": doc.get("lastTimestamp").isoformat() if doc.get("lastTimestamp") else None,
            "context": doc.get("context")
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
            "timestamp": doc["timestamp"].isoformat() if isinstance(doc["timestamp"], datetime) else str(doc["timestamp"]),
            "context": doc.get("context")
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

# --- 🟢 GET /details endpoint ---
@fastapi_app.get("/details")
async def get_details(client: str):
    """Fetch name and info for a given client number"""
    doc = await configs_collection.find_one({"clientNumber": client})
    if not doc:
        # If not found, return empty fields for frontend to fill
        return {"name": "", "info": ""}
    return {
        "name": doc.get("name", ""),
        "info": doc.get("info", "")
    }

# --- 🟢 PUT /details endpoint ---
@fastapi_app.put("/details")
async def update_details(request: Request):
    """Update or create client details"""
    body = await request.json()
    client_number = body.get("client")
    name = body.get("name", "")
    info = body.get("info", "")

    if not client_number:
        raise HTTPException(status_code=400, detail="Client number is required")

    # Upsert document (update if exists, insert if not)
    await configs_collection.update_one(
        {"clientNumber": client_number},
        {"$set": {"name": name, "info": info, "updatedAt": datetime.utcnow()}},
        upsert=True
    )

    print(f"✅ Updated details for {client_number}: name={name}, info={info}")
    return {"status": "success", "message": "Details updated"}


@fastapi_app.api_route("/", methods=["GET", "POST", "HEAD"])
async def root():
    return {"message": "Hello World"}

@fastapi_app.post("/send-bulk-email")
async def send_bulk_email_drive(drive_link: str = Form(...), sheet_name: str = Form("Sheet1")):
    """
    Send bulk emails from a public Google Sheet.
    - Required columns: subject | content | recipients
    - Optional: attachments | cc | bcc
    - attachments: comma-separated public Drive links
      * Empty attachments are skipped
      * Failed downloads of non-empty attachments count as a failure
    """
    try:
        # Convert Google Sheets link to CSV
        if "docs.google.com/spreadsheets" in drive_link:
            try:
                file_id = drive_link.split("/d/")[1].split("/")[0]
                safe_sheet_name = quote(sheet_name)
                drive_link = f"https://docs.google.com/spreadsheets/d/{file_id}/gviz/tq?tqx=out:csv&sheet={safe_sheet_name}"
                print(f"✅ Converted Google Sheet link: {drive_link}")
            except IndexError:
                raise HTTPException(status_code=400, detail="Invalid Google Sheets link")

        # Download CSV
        df = pd.read_csv(drive_link)
        print(f"✅ Loaded {len(df)} rows from Google Sheet")

        # Validate required columns
        required_columns = {"subject", "content", "recipients"}
        missing = required_columns - set(df.columns)
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

        EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
        EMAIL_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
        if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
            raise HTTPException(status_code=400, detail="Missing EMAIL_ADDRESS or EMAIL_APP_PASSWORD env vars")

        sent_count = 0
        failed = []

        async with httpx.AsyncClient() as client:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)

                for _, row in df.iterrows():
                    try:
                        msg = EmailMessage()
                        msg["Subject"] = str(row["subject"])
                        msg["From"] = EMAIL_ADDRESS

                        # Recipients
                        recipients = [r.strip() for r in str(row["recipients"]).split(",") if r.strip()]
                        msg["To"] = ", ".join(recipients)

                        # Optional CC/BCC
                        cc_list = [r.strip() for r in str(row.get("cc", "")).split(",") if r.strip()]
                        bcc_list = [r.strip() for r in str(row.get("bcc", "")).split(",") if r.strip()]
                        if cc_list:
                            msg["Cc"] = ", ".join(cc_list)

                        # Content
                        content = str(row["content"])
                        if "<" in content and ">" in content:
                            msg.add_alternative(content, subtype="html")
                        else:
                            msg.set_content(content)

                        # Attachments (Drive links)
                        # Attachments (Drive links)
                        attachments_col = row.get("attachments")
                        if pd.notna(attachments_col) and str(attachments_col).strip() != "":
                            for att_link in str(attachments_col).split(","):
                                att_link = att_link.strip()
                                if not att_link:
                                    continue
                                try:
                                    # --- Convert Google Drive share link to direct download link ---
                                    if "drive.google.com" in att_link:
                                        if "/file/d/" in att_link:
                                            file_id = att_link.split("/d/")[1].split("/")[0]
                                            att_link = f"https://drive.google.com/uc?export=download&id={file_id}"
                                        elif "id=" in att_link:
                                            file_id = att_link.split("id=")[1].split("&")[0]
                                            att_link = f"https://drive.google.com/uc?export=download&id={file_id}"
                                        # else leave as-is; may be invalid, will error out

                                    att_resp = await client.get(att_link, headers={"User-Agent": "Mozilla/5.0"},
                                                                follow_redirects=True)
                                    if att_resp.status_code != 200:
                                        raise Exception(f"Attachment download failed (status {att_resp.status_code})")

                                    # Extract filename from Content-Disposition
                                    cd = att_resp.headers.get("Content-Disposition")
                                    if cd and "filename=" in cd:
                                        filename = cd.split("filename=")[1].strip(' "')
                                    else:
                                        # fallback if header not present
                                        filename = att_link.split("/")[-1]

                                    ctype, encoding = mimetypes.guess_type(filename)
                                    if ctype is None or encoding is not None:
                                        ctype = "application/octet-stream"
                                    maintype, subtype = ctype.split("/", 1)

                                    msg.add_attachment(att_resp.content, maintype=maintype, subtype=subtype,
                                                       filename=filename)

                                except Exception as e:
                                    # Non-empty attachment failed → count as failure
                                    raise Exception(f"Attachment error for {att_link}: {e}")

                        smtp.send_message(msg, to_addrs=recipients + cc_list + bcc_list)
                        sent_count += 1
                        print(f"✅ Sent email to {recipients}")

                    except Exception as e:
                        print(f"❌ Failed to send email to {row.get('recipients')}: {e}")
                        failed.append({"recipients": row.get("recipients"), "error": str(e)})

        return {"status": "done", "sent": sent_count, "failed": failed}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def get_image_type(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return img.format.lower()  # 'jpeg', 'png', etc.
    except Exception:
        return None

def extract_csv_from_image(image_bytes: bytes) -> str:
    img_type = get_image_type(image_bytes)
    if not img_type:
        raise ValueError("Cannot determine image type")
    mime = f"image/{img_type}"

    encoded = {
        "mime_type": mime,
        "data": image_bytes
    }

    prompt = """
            Extract all transaction rows from this bank statement.
            Return ONLY CSV with headers:
            bank_name, value_date, transaction_date, debit_credit_flag, amount, description
            Dates must be YYYY-MM-DD. Do not add extra text.
            """
    response = gen_model.generate_content([prompt, encoded])
    return response.text.strip()

def submit_to_prop360(row: dict):
    payload = {
        "formId": "68c2eb885d8b5ec633d3be86",
        "indicator": "custom-a462rgbzo",
        "owner": "XkPoOtdoSxSe5CbRfK4zZBmaZnR2",
        "isPublic": False,
        "destructive": False,
        "data": {
            "field-1757605194423-ofbqnqfso": row.get("bank_name", ""),
            "field-1757605079803-icw8ykc19": row.get("value_date", ""),
            "field-1757605069078-p5plna7qr": row.get("transaction_date", ""),
            "field-1757605718340-ue95ozr9u": row.get("debit_credit_flag", ""),
            "field-1757605508754-uea4iadqd": row.get("amount", ""),
            "field-1758478917909-jqxoz2s3h": row.get("description"),
            "attachedForms": []
        }
    }

    r = httpx.post(
        PROP360_URL,
        json=payload,
        headers={"Authorization": f"Bearer {PROP_AUTH_TOKEN}"}
    )
    return r.status_code, r.text

def extract_folder_id(url: str):
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if not match:
        raise ValueError("Invalid public folder link")
    return match.group(1)

async def list_files_in_folder(folder_id: str):
    url = (
        "https://www.googleapis.com/drive/v3/files"
        f"?q='{folder_id}'+in+parents"
        "&fields=files(id,name,mimeType)"
        f"&key={GOOGLE_API_KEY}"
    )
    async with httpx.AsyncClient() as client:
        res = await client.get(url)
        return res.json().get("files", [])

@fastapi_app.post("/bank-statements/from-drive-folder")
async def process_from_drive_folder(folder_link: str = Body(..., embed=True)):
    folder_id = extract_folder_id(folder_link)

    files = await list_files_in_folder(folder_id)

    if not files:
        return {"processed": []}

    all_results = []

    async with httpx.AsyncClient() as client:
        for f in files:
            allowed_mimes = [
                "image/jpeg",
                "image/png",
                "image/jpg"
            ]

            if f["mimeType"] not in allowed_mimes:
                continue

            download_url = f"https://drive.google.com/uc?export=download&id={f['id']}"
            response = await client.get(download_url, follow_redirects=True)
            image_bytes = response.content

            csv_text = extract_csv_from_image(image_bytes)

            rows = csv.DictReader(csv_text.splitlines())
            next(rows, None)

            for row in rows:
                status, response = submit_to_prop360(row)
                all_results.append({
                    "file": f["name"],
                    "row": row,
                    "status": status,
                    "response": response
                })

    return {"processed": all_results}

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