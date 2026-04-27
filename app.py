import streamlit as st
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.title("🧠 Autism RAG Chatbot")

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    return embed_model, tokenizer, model

embed_model, tokenizer, llm_model = load_models()

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
        docs.append(f"""
Child age: {row['Age_Mons']} months
Eye contact issues: {yes_no(row['A2'])}
Social interaction issues: {yes_no(row['A3'])}
Repetitive behavior: {yes_no(row['A4'])}
Autism score: {row['Qchat-10-Score']}
Family history: {row['Family_mem_with_ASD']}
Result: {row['Class/ASD Traits']}
""")

    docs.extend([
        "Autism is a neurodevelopmental condition affecting communication.",
        "Early signs include delayed speech and poor eye contact.",
        "Therapies include speech and behavioral therapy.",
        "Early diagnosis improves outcomes significantly."
    ])

    return docs

docs = load_data()

# =========================
# BUILD INDEX
# =========================
@st.cache_resource
def build_index(docs):
    embeddings = embed_model.encode(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

index = build_index(docs)

# =========================
# RETRIEVE
# =========================
def retrieve(query, k=3):
    q_emb = embed_model.encode([query])
    _, idx = index.search(np.array(q_emb), k)
    return [docs[i] for i in idx[0]]

# =========================
# GENERATE RESPONSE
# =========================
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    
    outputs = llm_model.generate(
        **inputs,
        max_length=150,
        do_sample=True,
        temperature=0.7
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================
# CHATBOT
# =========================
def chatbot(query):
    if "autism" not in query.lower():
        return "⚠️ Ask autism-related questions only."

    context = "\n".join(retrieve(query))

    prompt = f"""
You are an autism expert assistant.

Context:
{context}

Question:
{query}

Answer clearly:
"""

    return generate_text(prompt)

# =========================
# UI
# =========================
user_input = st.text_input("💬 Ask about autism:")

if user_input:
    st.write("### 🧠 Answer")
    st.write(chatbot(user_input))
