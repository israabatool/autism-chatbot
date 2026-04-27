import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🧠 Autism Chatbot (Stable Cloud Version)")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("autism.csv")
    df.columns = df.columns.str.strip()

    def yn(v):
        return "Yes" if v == 1 else "No"

    docs = []

    for _, row in df.iterrows():
        docs.append(f"""
Age: {row['Age_Mons']} months
Eye contact issues: {yn(row['A2'])}
Social issues: {yn(row['A3'])}
Repetitive behavior: {yn(row['A4'])}
Score: {row['Qchat-10-Score']}
Result: {row['Class/ASD Traits']}
""")

    docs.extend([
        "Autism is a neurodevelopmental condition.",
        "Early signs include delayed speech and poor eye contact.",
        "Autism varies in severity.",
        "Therapies include speech and behavioral therapy.",
        "Early diagnosis improves outcomes."
    ])

    return docs

docs = load_data()

# =========================
# TF-IDF (LIGHTWEIGHT)
# =========================
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(docs)

def retrieve(query, k=3):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, doc_vectors).flatten()
    top_idx = sims.argsort()[-k:][::-1]
    return [docs[i] for i in top_idx]

# =========================
# CHATBOT
# =========================
def chatbot(query):
    if "autism" not in query.lower():
        return "Please ask autism-related questions."

    context = "\n".join(retrieve(query))

    return f"""
🧠 Answer:

{context}

💡 Autism affects communication, behavior, and social interaction.
"""

# =========================
# UI
# =========================
q = st.text_input("Ask about autism:")

if q:
    st.write(chatbot(q))
