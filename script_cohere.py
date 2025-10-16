import streamlit as st
import pandas as pd
import numpy as np
import cohere
from time import time

# --- Cohere Setup ---
COHERE_API_KEY = "43ne7RbinxXERVHw6P04l2ErDWbAZrEw9ZU6VlSq"
cohere_client = cohere.Client(COHERE_API_KEY)

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("faq_tr.csv")
    df = df.dropna(subset=["OLASI SORULAR - PINAR AGENT 35 YAŞINDA ATİNADA YAŞIYOR", "CEVAPLAR "])
    df = df.reset_index(drop=True)
    return df

# --- Build Embedding Index (Cohere only) ---
@st.cache_resource(show_spinner=False)
def build_index(df):
    st.write("📦 Encoding dataset with **Cohere Multilingual v3.0**...")
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
    st.success(f"✅ Encoded {len(texts)} entries in {elapsed:.2f} seconds")
    return embeddings, texts

# --- Search with Cohere Embeddings ---
def semantic_search(user_query, df, embeddings, texts, top_k=3, threshold=0.5):
    query_vec = cohere_client.embed(
        texts=[user_query],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    ).embeddings[0]

    query_vec = np.array(query_vec, dtype="float32")

    # Compute cosine similarity manually
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

#results[0][0] → question text
#results[0][1] → answer text ✅
#results[0][2] → similarity score


# --- RAG Generation (Cohere Command Model) ---
def generate_rag_response(user_query, results, chat_history):
    # ✅ Step 1: Check if there are good results from the dataset
    if results and results[0][2] > 0.5:
        context = "\n\n".join([f"Soru: {q}\nCevap: {a}" for q, a, _ in results])
    else:
        # No strong match found → return polite hardcoded response
        polite_reply = (
            "Bu konuda elimde net bir bilgi bulunmuyor. "
            "Size en doğru yanıtı verebilmem için lütfen soruyu biraz daha detaylandırır mısınız?"
        )
        return polite_reply, "⚠️ Veri kümesinde ilgili içerik bulunamadı."

    # ✅ Step 2: Use last few turns from chat history (for conversational continuity)
    if chat_history:
        history_str = "\n".join([f"Kullanıcı: {u}\nAsistan: {a}" for u, a, _ in chat_history[-3:]])
    else:
        history_str = ""

    # ✅ Step 3: Define the system prompt for the assistant's behavior
    system_prompt = (
        "Sen Türkçe konuşan profesyonel bir emlak danışmanısın. "
        "Verilen bağlamı ve konuşma geçmişini kullanarak "
        "kısa, net ve doğal bir yanıt ver. "
        "Eğer emin değilsen 'Bundan emin değilim.' de. "
        "Samimi, güven veren ve ikna edici bir üslup kullan."
        "Kendi bilgi bankanızdan cevap vermeyin. Herhangi bir bilgi için yalnızca bağlama güvenin. BU GERÇEKTEN ÖNEMLİ."
    )

    # ✅ Step 4: Combine everything into the model prompt
    user_prompt = (
        f"Geçmiş:\n{history_str}\n\n"
        f"Kullanıcının yeni sorusu: {user_query}\n\n"
        f"Bağlam:\n{context}"
    )

    # ✅ Step 5: Call Cohere only when relevant data is available
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

# --- Streamlit UI ---
st.set_page_config(page_title="Türkçe Chatbot 💬", page_icon="💬", layout="wide")
st.title("💬 Türkçe Chatbot – Cohere RAG Sürümü")

df = load_data()
embeddings, texts = build_index(df)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Sorunuzu yazın...") #Write your question

if user_query:
    with st.spinner("Yanıt hazırlanıyor..."):  # Response is being prepared
        answer, results = semantic_search(user_query, df, embeddings, texts)

        if answer is None:
            # ✅ No dataset match → return polite static response (no LLM call)
            rag_response = (
                "Bu konuda elimde net bir bilgi bulunmuyor. "
                "Ben yalnızca **Yunanistan Golden Visa** programı ile ilgili sorulara yardımcı olabiliyorum. 🇬🇷 "
                "Lütfen sorunuz bu konuyla ilgiliyse tekrar yazın. 😊"
            )
            context_used = "⚠️ Veri kümesinde ilgili içerik bulunamadı."
        else:
            # ✅ Dataset match → use RAG generation (calls LLM)
            rag_response, context_used = generate_rag_response(
                user_query, results, st.session_state.chat_history
            )

    st.session_state.chat_history.append((user_query, rag_response, context_used))

# --- Display Chat ---
for user_msg, bot_msg, ctx in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)
        if ctx:
            with st.expander("📚 Görülen Bağlam (Datasetten Alınan Bilgi)"):
                st.markdown(f"<div style='font-size:13px; white-space:pre-wrap;'>{ctx}</div>", unsafe_allow_html=True)
