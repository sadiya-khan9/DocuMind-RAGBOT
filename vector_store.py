from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


COLLECTION_NAME = "documind_documents"


def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="./local_bge_model",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )


def create_vector_store(
    documents,
    persist_directory="./chroma_db"
):

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    # Clear existing documents before rebuilding
    try:
        vector_store.delete_collection()
    except Exception:
        pass

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    ids = []

    for index, document in enumerate(documents):

        source = document.metadata.get(
            "source",
            "unknown"
        )

        page = document.metadata.get(
            "page",
            0
        )

        chunk_id = f"{source}_{page}_{index}"

        ids.append(chunk_id)

    vector_store.add_documents(
        documents=documents,
        ids=ids
    )

    return vector_store


def load_vector_store(
    persist_directory="./chroma_db"
):

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    return vector_store


def delete_document(
    filename,
    persist_directory="./chroma_db"
):

    vector_store = load_vector_store(
        persist_directory
    )

    vector_store.delete(
        where={
            "source": filename
        }
    )

    return vector_store