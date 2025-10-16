import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from time import time

# --- Cohere setup ---
try:
    import cohere
    cohere_client = cohere.Client("43ne7RbinxXERVHw6P04l2ErDWbAZrEw9ZU6VlSq")
except ImportError:
    cohere_client = None

# --- Model Options ---
MODEL_OPTIONS = {
    "MiniLM (Default)": "paraphrase-multilingual-MiniLM-L12-v2",
    "E5 Base": "intfloat/multilingual-e5-base",
    "Cohere Multilingual (API)": "cohere-multilingual-v3.0",
}

# --- Load Turkish FAQ Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("faq_tr.csv")
    df = df.dropna(subset=["OLASI SORULAR - PINAR AGENT 35 YAŞINDA ATİNADA YAŞIYOR", "CEVAPLAR "])
    df = df.reset_index(drop=True)
    return df

# --- Clean Texts ---
def clean_texts(series):
    texts = series.astype(str).replace(["nan", "NaN", "None"], "").fillna("")
    texts = [t.strip() for t in texts if t.strip() != ""]
    return texts

# --- Build FAISS Index ---
@st.cache_resource(show_spinner=False)
def build_index(df, model_name):
    st.write(f"📦 Encoding with **{model_name}**...")
    start = time()

    texts = clean_texts(df["OLASI SORULAR - PINAR AGENT 35 YAŞINDA ATİNADA YAŞIYOR"])

    if model_name == "cohere-multilingual-v3.0":
        if cohere_client is None:
            st.error("❌ Cohere API not available. Install with `pip install cohere` and add your API key.")
            return None, None, None

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
        model = None
    else:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings))
    elapsed = time() - start
    st.success(f"✅ Index built in {elapsed:.2f} seconds")

    return model, index, texts

# --- Search Function with confidence filtering ---
def get_answer(user_query, df, model, index, model_name, texts, top_k=3, threshold=0.45):
    if model_name == "cohere-multilingual-v3.0":
        query_vec = cohere_client.embed(
            texts=[user_query],
            model="embed-multilingual-v3.0",
            input_type="search_query"
        ).embeddings
        query_vec = np.array(query_vec, dtype="float32")
    else:
        query_vec = model.encode([user_query], normalize_embeddings=True)

    query_vec = np.nan_to_num(query_vec, nan=0.0, posinf=0.0, neginf=0.0)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for j, i in enumerate(indices[0]):
        q = df.iloc[i]["OLASI SORULAR - PINAR AGENT 35 YAŞINDA ATİNADA YAŞIYOR"]
        a = df.iloc[i]["CEVAPLAR "]
        s = float(scores[0][j])
        results.append((q, a, s))

    best_score = results[0][2] if results else 0.0
    if best_score < threshold:
        return None, results  # No confident dataset match
    return results[0][1], results

# --- RAG Generation (Multi-turn aware) ---
def generate_rag_response(user_query, results, chat_history):
    if cohere_client is None:
        return "❌ Cohere API not available."

    # Build context from retrieved docs
    if results and results[0][2] > 0.45:
        context = "\n\n".join([f"Soru: {q}\nCevap: {a}" for q, a, _ in results])
    else:
        context = "Veri kümesinde bu soruya doğrudan bir yanıt bulunamadı."

    # Build conversation history
    history_str = "\n".join([f"Kullanıcı: {u}\nAsistan: {a}" for u, a, _ in chat_history[-3:]])

    system_prompt = (
        "Sen Türkçe konuşan yardımcı bir asistansın. "
        "Konuşma geçmişini ve aşağıdaki bağlamı dikkate alarak, "
        "kullanıcının son sorusuna kısa, doğal ve doğru bir yanıt ver. "
        "Eğer emin değilsen 'Bu konuda emin değilim.' de. "
        "Yanıt profesyonel Türkçe olmalı."
    )

    user_prompt = f"Geçmiş:\n{history_str}\n\nKullanıcının yeni sorusu: {user_query}\n\nBağlam:\n{context}"

    try:
        response = cohere_client.chat(
            model="command-a-03-2025",
            message=user_prompt,
            preamble=system_prompt,
            temperature=0.6,
        )
        return response.text.strip(), context
    except Exception as e:
        return f"⚠️ Hata: {str(e)}", context

# --- Streamlit UI ---
st.set_page_config(page_title="Türkçe Chatbot 💬", page_icon="💬", layout="wide")
st.title("💬 Türkçe Chatbotu – Çoklu Tur (Cohere RAG)")

df = load_data()

model_choice = st.selectbox("Model Seç:", list(MODEL_OPTIONS.keys()))
model_name = MODEL_OPTIONS[model_choice]
model, index, texts = build_index(df, model_name)

# --- Initialize session state for conversation ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Sorunuzu yazın...")

if user_query:
    with st.spinner("Yanıt hazırlanıyor..."):
        answer, results = get_answer(user_query, df, model, index, model_name, texts)

        if answer is None:
            if cohere_client:
                rag_response = cohere_client.chat(
                    model="command-a-03-2025",
                    message=f"Kullanıcı şöyle dedi: '{user_query}'. "
                            "Bu konuda veri kümesinde bilgi yok. "
                            "Nazik, yönlendirici ve doğal bir Türkçe yanıt ver.",
                    temperature=0.7
                ).text.strip()
                context_used = "⚠️ Veri kümesinde ilgili içerik bulunamadı."
            else:
                rag_response = "Bu konuda emin değilim. Lütfen sorunuzu biraz daha açıklar mısınız?"
                context_used = ""
        else:
            rag_response, context_used = generate_rag_response(user_query, results, st.session_state.chat_history)

    st.session_state.chat_history.append((user_query, rag_response, context_used))

# --- Display full chat conversation ---
for user_msg, bot_msg, ctx in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)
        if ctx:
            with st.expander("📚 Görülen Bağlam (Datasetten Alınan Bilgi)"):
                st.markdown(f"<div style='font-size:13px; white-space:pre-wrap;'>{ctx}</div>", unsafe_allow_html=True)
