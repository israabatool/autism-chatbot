import streamlit as st
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# =========================
# APP TITLE
# =========================
st.title("🧠 Autism RAG Chatbot (LLM Powered)")

# =========================
# LOAD MODELS (SAFE)
# =========================
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    # FIX: use text-generation (most compatible on cloud)
    generator = pipeline(
        "text-generation",
        model="distilgpt2",
        max_length=200
    )

    return embed_model, generator

embed_model, generator = load_models()

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("autism.csv")
    df.columns = df.columns.str.strip()

    def yes_no(v):
        return "Yes" if v == 1 else "No"

    docs = []

    for _, row in df.iterrows():
        text = f"""
        Child age: {row['Age_Mons']} months
        Eye contact issues: {yes_no(row['A2'])}
        Social interaction issues: {yes_no(row['A3'])}
        Repetitive behavior: {yes_no(row['A4'])}
        Autism score: {row['Qchat-10-Score']}
        Family history: {row['Family_mem_with_ASD']}
        Result: {row['Class/ASD Traits']}
        """
        docs.append(text)

    # Add general autism knowledge
    docs.extend([
        "Autism is a neurodevelopmental condition affecting communication.",
        "Early signs include delayed speech and poor eye contact.",
        "Therapies include speech and behavioral therapy.",
        "Early diagnosis improves outcomes significantly."
    ])

    return docs

docs = load_data()

# =========================
# BUILD FAISS INDEX
# =========================
@st.cache_resource
def build_index(docs):
    embeddings = embed_model.encode(docs)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index

index = build_index(docs)

# =========================
# RETRIEVAL FUNCTION
# =========================
def retrieve(query, k=3):
    q_emb = embed_model.encode([query])
    _, idx = index.search(np.array(q_emb), k)
    return [docs[i] for i in idx[0]]

# =========================
# CHATBOT (RAG)
# =========================
def chatbot(query):

    if "autism" not in query.lower():
        return "⚠️ Please ask only autism-related questions."

    context = "\n".join(retrieve(query))

    prompt = f"""
You are a helpful autism assistant.

Context:
{context}

Question:
{query}

Answer in simple terms:
"""

    result = generator(prompt, max_length=200, do_sample=True)
    return result[0]['generated_text']

# =========================
# UI
# =========================
user_input = st.text_input("💬 Ask a question about autism:")

if user_input:
    response = chatbot(user_input)
    st.write("### 🧠 Answer")
    st.write(response)
