import os
import random
import asyncio
from time import time

import pandas as pd
import numpy as np
import openai
import requests
import httpx
from fastapi import FastAPI, Request
import uvicorn
from dotenv import load_dotenv
from urllib.parse import quote
from langdetect import detect

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
global_top_k = {"value": 5}
global_temperature = {"value": 0.2}
model_name = "text-embedding-3-small"


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


# --- Semantic Search ---
def semantic_search(user_query, df, embeddings, texts, model_name="text-embedding-3-small", top_k=2, threshold=0.5):
    print(f"\n🔎 Semantic search started for query: '{user_query}' with model: '{model_name}'")

    query_vec = openai.embeddings.create(
        input=[user_query],
        model=model_name
    ).data[0].embedding

    query_vec = np.array(query_vec, dtype="float32")
    sim_scores = np.dot(embeddings, query_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-10
    )

    top_idx = np.argsort(sim_scores)[::-1][:top_k]
    results = []

    print("📊 Top similarity results:")
    for i in top_idx:
        q = df.iloc[i]["questions"]
        a = df.iloc[i]["answers"]
        s = float(sim_scores[i])
        results.append((q, a, s))
        print(f"  → Q: {q[:80]}... | Score={s:.4f}")

    best_score = results[0][2] if results else 0.0
    print(f"🏁 Best score: {best_score:.4f} | Threshold: {threshold}")

    if best_score < threshold:
        print("⚠️ No match exceeded the threshold.\n")
        return None, results

    print(f"✅ Match found above threshold: '{results[0][0]}' (Score={best_score:.4f})\n")
    return results[0][1], results


# --- RAG Response Generation ---
def generate_rag_response(user_query, results, chat_history):
    print("🔧 Threshold used inside generate_rag_response:", global_threshold["value"])
    if results and results[0][2] > global_threshold["value"]:
        context = "\n\n".join([f"Soru: {q}\nCevap: {a}" for q, a, _ in results])
    else:
        polite_reply = (
            "Bu konuda elimde net bir bilgi bulunmuyor. "
            "Size en doğru yanıtı verebilmem için lütfen soruyu biraz daha detaylandırır mısınız?"
        )
        return polite_reply, "⚠️ Veri kümesinde ilgili içerik bulunamadı."

    history_str = (
        "\n".join([f"Kullanıcı: {u}\nAsistan: {a}" for u, a, _ in chat_history[-3:]])
        if chat_history else ""
    )

    system_prompt = f"{system_prompt_text}\n\n [IMPORTANT NOTE] Answer in the same language as of user query: {user_query}."

    user_prompt = (
        f"Geçmiş konuşma:\n{history_str}\n\n"
        f"Kullanıcının yeni sorusu: {user_query}\n\n"
        f"Bağlam (dataset'ten alınan bilgiler):\n{context}"
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=global_temperature["value"],
        )
        return response.choices[0].message.content.strip(), context
    except Exception as e:
        return f"⚠️ Hata: {str(e)}", context

# --- Initial Load ---
df = load_dataset_from_google_sheet(SHEET_ID)
embeddings, texts = build_index(df)
chat_sessions = {}


# --- FastAPI App ---
app = FastAPI()


@app.get("/webhook")
async def verify(request: Request):
    params = request.query_params
    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == VERIFY_TOKEN
    ):
        return int(params.get("hub.challenge"))
    return "Verification failed", 403


@app.post("/webhook")
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
        text = msg["text"]["body"].strip().lower()

        # --- Admin-only stop/start/refresh ---
        if from_number in ADMIN_NUMBERS:
            # Threshold update
            if text.startswith("threshold="):
                try:
                    new_threshold = float(text.split("=", 1)[1])
                    global_threshold["value"] = new_threshold
                    await send_whatsapp_message(from_number, f"✅ Threshold updated to {new_threshold}")
                    print(f"⚙️ Threshold updated to {new_threshold} by admin.")
                except ValueError:
                    await send_whatsapp_message(from_number, "⚠️ Invalid threshold format. Use like: threshold=0.35")
                return "THRESHOLD_UPDATED", 200
        
            # Top_k update
            if text.startswith("top_k="):
                try:
                    new_top_k = int(text.split("=", 1)[1])
                    global_top_k["value"] = new_top_k
                    await send_whatsapp_message(from_number, f"✅ top_k updated to {new_top_k}")
                    print(f"⚙️ top_k updated to {new_top_k} by admin.")
                except ValueError:
                    await send_whatsapp_message(from_number, "⚠️ Invalid top_k format. Use like: top_k=2")
                return "TOPK_UPDATED", 200

            if text.startswith("temperature="):
                try:
                    new_temperature = float(text.split("=", 1)[1])
                    global_temperature["value"] = new_temperature
                    await send_whatsapp_message(from_number, f"✅ temperature updated to {new_temperature}")
                    print(f"⚙️ temperature updated to {new_temperature} by admin.")
                except ValueError:
                    await send_whatsapp_message(from_number, "⚠️ Invalid temperature format. Use like: temperature=0.1")
                return "TEMPERATURE_UPDATED", 200

            if text == "status":
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

            if text == "prompt":
                prompt_message = (
                    f"📊 *Current RAG Prompt:*\n{system_prompt_text}"
                )
                await send_whatsapp_message(from_number, prompt_message)
                print(f"ℹ️ prompt requested by admin: {prompt_message}")
                return "PROMPT_SENT", 200

            if text == "stop":
                bot_active = False
                await send_whatsapp_message(from_number, "⏸ Bot paused globally.")
                print("🚫 Bot paused by admin.")
                return "BOT_PAUSED", 200
        
            if text == "start":
                bot_active = True
                await send_whatsapp_message(from_number, "▶️ Bot resumed globally.")
                print("✅ Bot resumed by admin.")
                return "BOT_RESUMED", 200
        
            if text.startswith("refresh"):
                print("🔄 Admin requested dataset refresh...")
                df = load_dataset_from_google_sheet(SHEET_ID)

                # Determine which model to use
                if "2" in text:
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
        answer, results = semantic_search(text, df, embeddings, texts, model_name=model_name, top_k=global_top_k["value"], threshold=global_threshold["value"])


        if answer is None:
            print("🔍 No strong match found. Top results:")
            for q, a, s in results:
                print(f"  → {q} (score={s:.2f})")
            rag_response = (
                "Bu konuda elimde net bir bilgi bulunmuyor. "
                "Ben yalnızca **Yunanistan Golden Visa** programı ile ilgili sorulara yardımcı olabiliyorum. 🇬🇷 "
                "Lütfen sorunuz bu konuyla ilgiliyse tekrar yazın. 😊"
            )
            context_used = "⚠️ Veri kümesinde ilgili içerik bulunamadı."
        else:
            rag_response, context_used = generate_rag_response(
                text, results, chat_history
            )

        chat_history.append((text, rag_response, context_used))
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
        print("📩 Output message:", payload)
        requests.post(url, headers=headers, json=payload)

    except Exception as e:
        print("⚠️ Error handling message:", e)

    return "EVENT_RECEIVED", 200


@app.api_route("/", methods=["GET", "POST", "HEAD"])
async def root():
    return {"message": "Hello World"}


@app.post("/reload")
async def reload_data():
    global df, embeddings, texts
    df = load_dataset_from_google_sheet(SHEET_ID)
    embeddings, texts = build_index(df)
    return {"status": "✅ Dataset reloaded successfully"}


# --- Helper for sending WhatsApp messages ---
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

    print("📤 Sending message:", payload)
    async with httpx.AsyncClient() as client:
        await client.post(url, headers=headers, json=payload)


if __name__ == "__main__":
    print("🚀 WhatsApp bot running at /webhook")
    uvicorn.run(app, host="0.0.0.0", port=8000)
