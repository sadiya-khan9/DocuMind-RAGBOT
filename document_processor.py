import os
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader


def load_documents(directory):
    documents = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if filename.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())

        elif filename.lower().endswith(".pdf"):
            try:
                images = convert_from_path(file_path, dpi=200)

                for page_number, image in enumerate(images, start=1):
                    text = pytesseract.image_to_string(image)

                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": filename,
                                    "page": page_number
                                }
                            )
                        )

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return documents


def chunk_documents(documents):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_documents(documents)