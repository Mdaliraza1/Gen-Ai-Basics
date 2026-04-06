# rag_pipeline.py

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
import os


# 1. Load multiple file types
def load_documents(path="data"):
    loader = DirectoryLoader(
        path,
        glob="**/*.*",
        loader_cls=UnstructuredFileLoader
    )
    return loader.load()


# 2. Split documents
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)


# 3. Create or load vector DB
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore


# 4. LLM
def get_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5}
    )


# 5. Ask question
def ask_question(query, retriever, llm):
    docs = retriever.get_relevant_documents(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer based on the context below:

    Context:
    {context}

    Question:
    {query}
    """

    return llm.invoke(prompt)