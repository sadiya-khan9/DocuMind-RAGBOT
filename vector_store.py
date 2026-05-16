from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def create_vector_store(documents, persist_directory="./chroma_db"):
    embeddings = HuggingFaceEmbeddings(
        model_name="./local_bge_model",  # Use the locally saved BGE model
        model_kwargs={"device": "cpu"},  # Use 'cuda' if you have a GPU
        encode_kwargs={"normalize_embeddings": True},  # Crucial for BGE accuracy
    )
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vector_store


def load_vector_store(persist_directory="./chroma_db"):
    embeddings = HuggingFaceEmbeddings(
        model_name="./local_bge_model",  # Use the locally saved BGE model
        model_kwargs={"device": "cpu"},  # Use 'cuda' if you have a GPU
        encode_kwargs={"normalize_embeddings": True},  # Crucial for BGE accuracy
    )
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    return vector_store
