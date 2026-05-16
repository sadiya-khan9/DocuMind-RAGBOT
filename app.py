import os
import streamlit as st
from dotenv import load_dotenv
from document_processor import load_documents, chunk_documents
from vector_store import create_vector_store, load_vector_store
from rag_chain import create_rag_chain

load_dotenv()

DB_PATH = "./chroma_db"
DOCS_DIR = "./knowledge_base"

st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
st.title("Context-Aware RAG Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


@st.cache_resource
def initialize_chain():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found in environment variables or .env file")
        return None

    if os.path.exists(DB_PATH):
        vector_store = load_vector_store(DB_PATH)
    else:
        if not os.path.exists(DOCS_DIR):
            os.makedirs(DOCS_DIR)
        if os.listdir(DOCS_DIR):
            documents = load_documents(DOCS_DIR)
            chunks = chunk_documents(documents)
            vector_store = create_vector_store(chunks, DB_PATH)
        else:
            st.warning(f"No documents found in {DOCS_DIR}. Add PDF or TXT files.")
            return None

    return create_rag_chain(vector_store, api_key)


if st.session_state.rag_chain is None:
    st.session_state.rag_chain = initialize_chain()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.rag_chain:
            response = st.session_state.rag_chain({
                "input": prompt,
                "chat_history": st.session_state.chat_history,
            })
            answer = response["answer"]
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        else:
            st.error("Chatbot not initialized. Please check your API key and documents.")

if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

if st.sidebar.button("Rebuild Vector Store"):
    if os.path.exists(DB_PATH):
        import shutil
        shutil.rmtree(DB_PATH)
    st.session_state.rag_chain = None
    st.session_state.chat_history = []
    st.rerun()
