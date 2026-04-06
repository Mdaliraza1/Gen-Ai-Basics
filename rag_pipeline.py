# rag_pipeline.py

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# 1. Load multiple file types

def load_documents(path="data"):
    docs = []

    txt_loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)

    docs.extend(txt_loader.load())
    docs.extend(pdf_loader.load())

    return docs
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
        return FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore


# 4. LLM
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",   # fast + cheap + good
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    return HuggingFacePipeline(pipeline=pipe)


# 5. Ask question
def ask_question(query, retriever, llm):
    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content[:300] for doc in docs[:2]])

    prompt = f"""
Context:
{context}

Question: {query}

Answer in one sentence:"""

    response = llm.invoke(prompt)
    answer = response.content

    return answer