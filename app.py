import numpy as np
import pandas as pd
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline

st.title("🧠 Autism Support Chatbot")

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256
    )
    return embed_model, generator

embed_model, generator = load_models()

@st.cache_data
def load_data():
    df = pd.read_csv("autism.csv")
    df.columns = df.columns.str.strip()

    def yes_no(val):
        return "Yes" if val == 1 else "No"

    docs = []
    for _, row in df.iterrows():
        text = f"""
        Age: {row['Age_Mons']} months
        Eye contact: {yes_no(row['A2'])}
        Social issues: {yes_no(row['A3'])}
        Repetitive behavior: {yes_no(row['A4'])}
        Result: {row['Class/ASD Traits']}
        """
        docs.append(text)

    docs.extend([
        "Autism is a neurodevelopmental condition.",
        "Early signs include delayed speech and poor eye contact.",
        "Therapies include speech and behavioral therapy."
    ])

    return docs

docs = load_data()

@st.cache_resource
def build_index(docs):
    embeddings = embed_model.encode(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

index = build_index(docs)

def retrieve(query, k=3):
    q_emb = embed_model.encode([query])
    _, indices = index.search(np.array(q_emb), k)
    return [docs[i] for i in indices[0]]

def chatbot(query):
    context = "\n".join(retrieve(query))
    prompt = f"""
    Answer using context:
    {context}

    Question: {query}
    """
    return generator(prompt)[0]['generated_text']

query = st.text_input("Ask about autism:")

if query:
    st.write(chatbot(query))