# app.py

import streamlit as st
from rag_pipeline import (
    load_documents,
    split_documents,
    get_vectorstore,
    get_llm,
    ask_question
)

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("🤖 GenAI RAG Chatbot")
st.write("Ask questions from your uploaded documents")

# Load pipeline (only once)
@st.cache_resource
def setup():
    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = get_vectorstore(chunks)
    retriever = vectorstore.as_retriever()
    llm = get_llm()
    return retriever, llm

retriever, llm = setup()

# Chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.text_input("Ask a question:")

if query:
    answer = ask_question(query, retriever, llm)

    st.session_state.chat.append(("You", query))
    st.session_state.chat.append(("AI", answer))

# Display chat
for role, msg in st.session_state.chat:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 AI:** {msg}")