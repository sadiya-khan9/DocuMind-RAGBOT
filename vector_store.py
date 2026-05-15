import os
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def create_vector_store(documents, persist_directory="./chroma_db"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vector_store


def load_vector_store(persist_directory="./chroma_db"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    return vector_store
