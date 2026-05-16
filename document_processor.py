import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
        elif filename.endswith('.txt'):
            try:
                loader = TextLoader(filepath, encoding='utf-8')
                documents.extend(loader.load())
            except Exception:
                loader = TextLoader(filepath, encoding='latin-1')
                documents.extend(loader.load())
    return documents


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)
