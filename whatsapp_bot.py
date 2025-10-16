import pandas as pd
import numpy as np
import cohere
from time import time
from fastapi import FastAPI, Request
import uvicorn
import requests
from dotenv import load_dotenv
import os
import cohere

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
cohere_client = cohere.Client(COHERE_API_KEY)

WHATSAPP_ACCESS_TOKEN = "EAAJ6jOg3B0QBPnxguL4NLI0MvnQAcVMvBObceiNNDLEa62J7Gbs42HbKFoWFKG0yuXEDKwzFMzenZB1shtST5Ypyl16fiCZCG5WhDh7GtzqlveZCQQ5VKFRVmXZBDIKZA9JM9AoocCBPRFNNmUkZCb3PJu0gDzBw9bumSthl99gkLtY1nQ1sHBthjts11mB8f4oqM92yyUsgcXM7GiEZBruZA01CSrTCZBLajshq87YgZC"
PHONE_NUMBER_ID = "817065164826591"

# --- Load Dataset ---
def load_data():
    df = pd.read_csv("faq_tr.csv")
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


# --- Search with Cohere Embeddings ---
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


# --- RAG Generation (Cohere Command Model) ---
def generate_rag_response(user_query, results, chat_history):
    if results and results[0][2] > 0.5:
        context = "\n\n".join([f"Soru: {q}\nCevap: {a}" for q, a, _ in results])
    else:
        polite_reply = (
            "Bu konuda elimde net bir bilgi bulunmuyor. "
            "Size en doğru yanıtı verebilmem için lütfen soruyu biraz daha detaylandırır mısınız?"
        )
        return polite_reply, "⚠️ Veri kümesinde ilgili içerik bulunamadı."

    if chat_history:
        history_str = "\n".join([f"Kullanıcı: {u}\nAsistan: {a}" for u, a, _ in chat_history[-3:]])
    else:
        history_str = ""

    system_prompt = (
        "Sen Türkçe konuşan profesyonel bir emlak danışmanısın. "
        "Verilen bağlamı ve konuşma geçmişini kullanarak "
        "kısa, net ve doğal bir yanıt ver. "
        "Eğer emin değilsen 'Bundan emin değilim.' de. "
        "Samimi, güven veren ve ikna edici bir üslup kullan."
        "Kendi bilgi bankanızdan cevap vermeyin. Herhangi bir bilgi için yalnızca bağlama güvenin. BU GERÇEKTEN ÖNEMLİ."
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


# --- Load everything once ---
df = load_data()
embeddings, texts = build_index(df)
chat_sessions = {}  # store conversation history per WhatsApp user


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
    data = await request.json()
    print("📩 Incoming message:", data)

    try:
        entry = data["entry"][0]["changes"][0]["value"]
        messages = entry.get("messages")
        if messages:
            msg = messages[0]
            from_number = msg["from"]
            text = msg["text"]["body"]

            # retrieve chat history for this user
            chat_history = chat_sessions.get(from_number, [])

            # same logic as before
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

            # save history
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
            requests.post(url, headers=headers, json=payload)

    except Exception as e:
        print("⚠️ Error handling message:", e)

    return "EVENT_RECEIVED", 200


if __name__ == "__main__":
    print("🚀 WhatsApp bot running at /webhook")
    uvicorn.run(app, host="0.0.0.0", port=8000)
