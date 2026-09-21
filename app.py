import os
import streamlit as st
from dotenv import load_dotenv

from document_processor import load_documents, chunk_documents
from vector_store import (
    create_vector_store,
    load_vector_store,
    delete_document
)
from rag_chain import create_rag_chain

from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 15% 20%,
            rgba(99, 102, 241, 0.18),
            transparent 28%
        ),
        radial-gradient(
            circle at 85% 10%,
            rgba(168, 85, 247, 0.15),
            transparent 25%
        ),
        radial-gradient(
            circle at 70% 80%,
            rgba(59, 130, 246, 0.10),
            transparent 25%
        ),
        #080b14;

    color: #f8fafc;
}


.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}


/* ---------------- BRAND ---------------- */

.brand {
    text-align: left;
    margin-top: 70px;
    margin-left: 25px;
    animation: fadeUp 0.8s ease;
}


.brand-icon {
    width: 62px;
    height: 62px;

    margin: 0;

    border-radius: 20px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 30px;

    background: linear-gradient(
        135deg,
        #6366f1,
        #8b5cf6,
        #3b82f6
    );

    box-shadow:
        0 0 35px rgba(99,102,241,0.35),
        0 0 80px rgba(139,92,246,0.12);

    animation: float 4s ease-in-out infinite;
}


.brand-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -2px;
    margin-top: 18px;

    background: linear-gradient(
        90deg,
        #ffffff,
        #c4b5fd,
        #93c5fd
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}


.brand-subtitle {
    color: #94a3b8;
    font-size: 15px;
    margin-top: 8px;
}


/* ---------------- WELCOME ---------------- */

.welcome-card {
    margin: 45px 25px 25px 25px;
    max-width: 850px;

    padding: 30px;

    border-radius: 25px;

    text-align: left;

    background:
        linear-gradient(
            135deg,
            rgba(99,102,241,0.10),
            rgba(139,92,246,0.06)
        );

    border: 1px solid rgba(139,92,246,0.20);

    box-shadow:
        0 20px 60px rgba(0,0,0,0.20);

    animation: fadeUp 0.9s ease;
}


.welcome-title {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 10px;
}


.welcome-text {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.7;
}


/* ---------------- SUGGESTIONS ---------------- */

.suggestion-title {
    text-align: left;

    color: #64748b;

    font-size: 12px;

    margin-top: 25px;
    margin-bottom: 12px;
    margin-left: 25px;

    letter-spacing: 0.5px;
}


/* ---------------- CHAT ---------------- */

.chat-user {
    background: linear-gradient(
        135deg,
        rgba(99,102,241,0.25),
        rgba(139,92,246,0.18)
    );

    border: 1px solid rgba(139,92,246,0.25);

    padding: 15px 18px;

    border-radius: 18px;

    margin: 15px 0 10px auto;

    max-width: 75%;
}


.chat-ai {
    background: rgba(255,255,255,0.04);

    border: 1px solid rgba(255,255,255,0.08);

    padding: 18px;

    border-radius: 18px;

    margin: 10px auto 15px 0;

    max-width: 85%;

    line-height: 1.7;
}


/* ---------------- SIDEBAR ---------------- */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #0b0f1a,
            #080b14
        );

    border-right: 1px solid rgba(255,255,255,0.06);
}


.sidebar-title {
    font-size: 22px;
    font-weight: 800;

    background: linear-gradient(
        90deg,
        #c4b5fd,
        #93c5fd
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}


.sidebar-subtitle {
    color: #64748b;

    font-size: 12px;

    margin-bottom: 25px;
}


.document-card {
    padding: 12px;

    margin: 8px 0;

    border-radius: 12px;

    background: rgba(255,255,255,0.04);

    border: 1px solid rgba(255,255,255,0.06);

    font-size: 13px;

    color: #cbd5e1;
}


/* ---------------- BUTTONS ---------------- */

.stButton > button {

    border-radius: 12px;

    border: 1px solid rgba(139,92,246,0.20);

    background: rgba(255,255,255,0.04);

    color: #e2e8f0;

    transition: all 0.25s ease;
}


.stButton > button:hover {

    border-color: rgba(139,92,246,0.60);

    background: rgba(139,92,246,0.12);

    transform: translateY(-2px);

    box-shadow:
        0 8px 25px rgba(99,102,241,0.15);
}


/* ---------------- ANIMATIONS ---------------- */

@keyframes fadeUp {

    from {
        opacity: 0;
        transform: translateY(15px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }

}


@keyframes float {

    0%, 100% {
        transform: translateY(0);
    }

    50% {
        transform: translateY(-7px);
    }

}


/* ---------------- FILE UPLOADER ---------------- */

[data-testid="stFileUploader"] {

    background: rgba(255,255,255,0.03);

    border-radius: 15px;

    padding: 8px;
}


/* ---------------- CHAT INPUT ---------------- */

[data-testid="stChatInput"] {
    border-radius: 18px;
}


/* ---------------- SCROLLBAR ---------------- */

::-webkit-scrollbar {
    width: 6px;
}


::-webkit-scrollbar-track {
    background: #080b14;
}


::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False


if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ---------------------------------------------------------
# INITIALIZE RAG
# ---------------------------------------------------------

def initialize_chain():

    if not os.getenv("GOOGLE_API_KEY"):

        st.error(
            "GOOGLE_API_KEY not found in .env"
        )

        return None


    knowledge_base = "./knowledge_base"

    chroma_db = "./chroma_db"


    os.makedirs(
        knowledge_base,
        exist_ok=True
    )


    try:

        if os.path.exists(chroma_db):

            vector_store = load_vector_store(
                chroma_db
            )

            return create_rag_chain(
                vector_store
            )


        documents = load_documents(
            knowledge_base
        )


        if not documents:

            return None


        chunks = chunk_documents(
            documents
        )


        vector_store = create_vector_store(
            chunks,
            chroma_db
        )


        return create_rag_chain(
            vector_store
        )


    except Exception as e:

        st.error(
            f"Error initializing chatbot: {e}"
        )

        return None


# ---------------------------------------------------------
# BUILD CHAT HISTORY
# ---------------------------------------------------------

def build_chat_history():

    messages = []


    for role, message in st.session_state.chat_history:

        if role == "user":

            messages.append(
                HumanMessage(
                    content=message
                )
            )


        elif role == "assistant":

            messages.append(
                AIMessage(
                    content=message
                )
            )


    return messages


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">✦ DocuMind AI</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="sidebar-subtitle">'
        'Your intelligent document assistant'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # UPLOAD
    # -----------------------------------------------------

    st.markdown(
        "### 📂 Add Documents"
    )


    uploaded_file = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        label_visibility="collapsed"
    )


    if uploaded_file is not None:

        os.makedirs(
            "knowledge_base",
            exist_ok=True
        )


        file_path = os.path.join(
            "knowledge_base",
            uploaded_file.name
        )


        with open(
            file_path,
            "wb"
        ) as f:

            f.write(
                uploaded_file.getbuffer()
            )


        st.success(
            "Document uploaded successfully."
        )


    # -----------------------------------------------------
    # PROCESS
    # -----------------------------------------------------

    if st.button(
        "⚡ Process Documents",
        use_container_width=True
    ):

        with st.spinner(
            "Processing documents..."
        ):

            try:

                documents = load_documents(
                    "./knowledge_base"
                )


                if not documents:

                    st.warning(
                        "No readable documents found."
                    )


                else:

                    chunks = chunk_documents(
                        documents
                    )


                    vector_store = create_vector_store(
                        chunks,
                        "./chroma_db"
                    )


                    st.session_state.rag_chain = (
                        create_rag_chain(
                            vector_store
                        )
                    )


                    st.session_state.documents_loaded = True


                    st.session_state.chat_history = []


                    st.success(
                        "Documents processed successfully! 🎉"
                    )


            except Exception as e:

                st.error(
                    f"Processing error: {e}"
                )


    st.divider()


    # -----------------------------------------------------
    # DOCUMENT LIST
    # -----------------------------------------------------

    st.markdown(
        "### 📚 Your Documents"
    )


    knowledge_base = "./knowledge_base"


    if os.path.exists(
        knowledge_base
    ):

        files = [
            f
            for f in os.listdir(
                knowledge_base
            )
            if f.lower().endswith(
                (".pdf", ".txt")
            )
        ]


        if files:

            for file in files:

                col1, col2 = st.columns(
                    [5, 1]
                )


                with col1:

                    st.markdown(
                        f"""
                        <div class="document-card">
                            📄 {file}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                with col2:

                    delete_key = (
                        f"delete_{file}"
                    )


                    if st.button(
                        "🗑️",
                        key=delete_key,
                        help=f"Delete {file}"
                    ):

                        try:

                            # Delete physical file
                            file_path = os.path.join(
                                knowledge_base,
                                file
                            )


                            if os.path.exists(
                                file_path
                            ):

                                os.remove(
                                    file_path
                                )


                            # Delete embeddings
                            if os.path.exists(
                                "./chroma_db"
                            ):

                                delete_document(
                                    file,
                                    "./chroma_db"
                                )


                            # Rebuild RAG chain
                            if os.path.exists(
                                "./chroma_db"
                            ):

                                vector_store = (
                                    load_vector_store(
                                        "./chroma_db"
                                    )
                                )


                                st.session_state.rag_chain = (
                                    create_rag_chain(
                                        vector_store
                                    )
                                )

                            else:

                                st.session_state.rag_chain = None


                            # Clear current chat
                            st.session_state.chat_history = []


                            st.success(
                                f"{file} deleted successfully."
                            )


                            st.rerun()


                        except Exception as e:

                            st.error(
                                f"Delete error: {e}"
                            )


        else:

            st.caption(
                "No documents uploaded yet."
            )


    st.divider()


    # -----------------------------------------------------
    # CHAT CONTROLS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "＋ New Chat",
            use_container_width=True
        ):

            st.session_state.chat_history = []

            st.rerun()


    with col2:

        if st.button(
            "Clear",
            use_container_width=True
        ):

            st.session_state.chat_history = []

            st.rerun()


# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------

st.markdown(
    "## ✦ DocuMind AI"
)


st.markdown(
    "Ask questions. Explore documents. Get answers."
)


# ---------------------------------------------------------
# LOAD EXISTING VECTOR STORE
# ---------------------------------------------------------

if (
    st.session_state.rag_chain is None
    and os.path.exists("./chroma_db")
):

    try:

        st.session_state.rag_chain = (
            initialize_chain()
        )


        if st.session_state.rag_chain:

            st.session_state.documents_loaded = True


    except Exception:

        pass


# ---------------------------------------------------------
# WELCOME SCREEN
# ---------------------------------------------------------

if len(
    st.session_state.chat_history
) == 0:

    st.markdown(
        "### 👋 Hey there!"
    )


    st.markdown(
        """
        How are you doing today?

        I'm ready to help you explore your documents.

        **What would you like to know?**
        """
    )


    st.markdown(
        "##### TRY ASKING"
    )


    # -----------------------------------------------------
    # SUGGESTION BUTTONS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "📄 Summarize my document",
            use_container_width=True
        ):

            st.session_state.pending_prompt = (
                "Summarize the document."
            )

            st.rerun()


        if st.button(
            "🔑 What are the key points?",
            use_container_width=True
        ):

            st.session_state.pending_prompt = (
                "What are the key points from the document?"
            )

            st.rerun()


    with col2:

        if st.button(
            "💡 Explain an important concept",
            use_container_width=True
        ):

            st.session_state.pending_prompt = (
                "Explain the most important concept in the document."
            )

            st.rerun()


        if st.button(
            "💬 Ask anything",
            use_container_width=True
        ):

            st.session_state.pending_prompt = (
                "What can you tell me about this document?"
            )

            st.rerun()


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

for role, message in st.session_state.chat_history:

    if role == "user":

        st.markdown(
            f"""
            <div class="chat-user">
                {message}
            </div>
            """,
            unsafe_allow_html=True
        )


    else:

        st.markdown(
            f"""
            <div class="chat-ai">
                {message}
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------------------------------------------------
# PROCESS PENDING SUGGESTION
# ---------------------------------------------------------

if st.session_state.pending_prompt is not None:

    prompt = st.session_state.pending_prompt


    st.session_state.pending_prompt = None


    if st.session_state.rag_chain is None:

        st.warning(
            "Please upload and process a document first."
        )


    else:

        st.session_state.chat_history.append(
            ("user", prompt)
        )


        with st.spinner(
            "Thinking..."
        ):

            try:

                chat_history = build_chat_history()


                # Remove current question from history
                chat_history = chat_history[:-1]


                response = st.session_state.rag_chain(
                    {
                        "input": prompt,
                        "chat_history": chat_history
                    }
                )


                if isinstance(
                    response,
                    dict
                ):

                    answer = response.get(
                        "answer",
                        str(response)
                    )


                else:

                    answer = str(
                        response
                    )


                st.session_state.chat_history.append(
                    ("assistant", answer)
                )


            except Exception as e:

                st.session_state.chat_history.append(
                    (
                        "assistant",
                        f"Sorry, something went wrong: {e}"
                    )
                )


        st.rerun()


# ---------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------

prompt = st.chat_input(
    "Ask something about your documents..."
)


if prompt:

    st.session_state.chat_history.append(
        ("user", prompt)
    )


    if st.session_state.rag_chain is None:

        st.session_state.chat_history.append(
            (
                "assistant",
                "Please upload and process a document first. 📄"
            )
        )

        st.rerun()


    with st.spinner(
        "Thinking..."
    ):

        try:

            chat_history = build_chat_history()


            # Remove current question from history
            chat_history = chat_history[:-1]


            response = st.session_state.rag_chain(
                {
                    "input": prompt,
                    "chat_history": chat_history
                }
            )


            if isinstance(
                response,
                dict
            ):

                answer = response.get(
                    "answer",
                    str(response)
                )


            else:

                answer = str(
                    response
                )


            st.session_state.chat_history.append(
                ("assistant", answer)
            )


        except Exception as e:

            st.session_state.chat_history.append(
                (
                    "assistant",
                    f"Sorry, something went wrong: {e}"
                )
            )


    st.rerun() 