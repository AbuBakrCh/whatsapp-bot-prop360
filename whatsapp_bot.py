import os
import random
import asyncio
from time import time

import pandas as pd
import numpy as np
import cohere
import requests
import httpx
from fastapi import FastAPI, Request
import uvicorn
from dotenv import load_dotenv

# --- Load Environment ---
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
cohere_client = cohere.Client(COHERE_API_KEY)

WHATSAPP_ACCESS_TOKEN = "EAAQYybD4vYYBPkO0ZCsdI7j4i8qPwkc0p6yyqcJILxYQkrj3MB9PuHXHa7ZAUCaWJLocryOdfHKkCV23LZAWxlMWFxzu9mUCQQhAJugijcTFkcNMzfCaPbT6hDqyaW4xkjinvtMxziTVhMiOXBOVMHe64OrRQEahCZBzENChcAeZAdug9sZBaHltSAYbSbDNdkQgZDZD"
PHONE_NUMBER_ID = "836070512923409"

# --- Google Drive CSV File ---
DRIVE_FILE_ID = "1M1-J99G936xLo8m8rWXvcvtxpj1E8ZuC"
CSV_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
DATASET_PATH = "faq_tr.csv"

ADMIN_NUMBERS = ["306980102740", "923244181389"]

bot_active = True

# --- Download CSV from Google Drive ---
def download_csv():
    print("📥 Downloading dataset from Google Drive...")
    response = requests.get(CSV_URL)
    response.raise_for_status()
    with open(DATASET_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Dataset downloaded successfully")

# --- Load Dataset ---
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        download_csv()
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["OLASI SORULAR - PINAR AGENT 35 YAŞINDA ATİNADA YAŞIYOR", "CEVAPLAR "])
    df = df.reset_index(drop=True)
    return df

# --- Build Embedding Index (Cohere only) ---
def build_index(df):
    print("📦 Encoding dataset with Cohere Multilingual v3.0...")
    start = time()

    texts = df["OLASI SORULAR - PINAR AGENT 35 YAŞINDA ATİNADA YAŞIYOR"].astype(str).tolist()
    all_embeddings = []

    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = cohere_client.embed(
            texts=batch,
            model="embed-multilingual-v3.0",
            input_type="search_document"
        )
        all_embeddings.extend(response.embeddings)

    embeddings = np.array(all_embeddings, dtype="float32")
    elapsed = time() - start
    print(f"✅ Encoded {len(texts)} entries in {elapsed:.2f} seconds")
    return embeddings, texts

# --- Semantic Search ---
def semantic_search(user_query, df, embeddings, texts, top_k=3, threshold=0.5):
    query_vec = cohere_client.embed(
        texts=[user_query],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    ).embeddings[0]

    query_vec = np.array(query_vec, dtype="float32")
    sim_scores = np.dot(embeddings, query_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-10
    )

    top_idx = np.argsort(sim_scores)[::-1][:top_k]
    results = []
    for i in top_idx:
        q = df.iloc[i]["OLASI SORULAR - PINAR AGENT 35 YAŞINDA ATİNADA YAŞIYOR"]
        a = df.iloc[i]["CEVAPLAR "]
        s = float(sim_scores[i])
        results.append((q, a, s))

    best_score = results[0][2] if results else 0.0
    if best_score < threshold:
        return None, results
    return results[0][1], results

# --- RAG Response Generation ---
def generate_rag_response(user_query, results, chat_history):
    if results and results[0][2] > 0.5:
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

    system_prompt = (
        "Sen Türkçe konuşan, profesyonel ama samimi bir emlak danışmanısın. "
        "Yanıtlarında doğal, içten ve insana benzeyen bir dil kullan. "
        "Cümlelerini kısa tut — genellikle birkaç kelime ya da tek bir kısa cümle kadar. "
        "Yapay zekâ gibi değil, bir insan gibi konuş: günlük kelimeler, doğal ifadeler, sade yazım. "
        "Verilen bağlam ve konuşma geçmişine dayanarak yanıt ver. "
        "Eğer emin değilsen 'Bundan emin değilim.' de. "
        "Kendi bilgi bankanı kullanma; yalnızca verilen bağlama güven. "
        "Bu çok önemli — bağlam dışında tahmin yürütme veya yeni bilgi üretme."
    )

    user_prompt = (
        f"Geçmiş:\n{history_str}\n\n"
        f"Kullanıcının yeni sorusu: {user_query}\n\n"
        f"Bağlam:\n{context}"
    )

    try:
        response = cohere_client.chat(
            model="command-a-03-2025",
            message=user_prompt,
            preamble=system_prompt,
            temperature=0.01,
        )
        return response.text.strip(), context
    except Exception as e:
        return f"⚠️ Hata: {str(e)}", context

# --- Initial Load ---
download_csv()
df = load_dataset()
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

        # --- Admin-only stop/start commands ---
        if from_number in ADMIN_NUMBERS:
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

            if text == "refresh":
                print("🔄 Admin requested dataset refresh...")
                download_csv()
                df = load_dataset()
                embeddings, texts = build_index(df)
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
        answer, results = semantic_search(text, df, embeddings, texts)

        if answer is None:
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
    download_csv()
    df = load_dataset()
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
