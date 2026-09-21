# Context-Aware Chatbot Using RAG

A conversational chatbot that remembers context and retrieves external information using Retrieval-Augmented Generation (RAG) with LangChain, ChromaDB, local BGE embeddings, and Google Gemini 2.5 Flash.

## Features

- Context-aware conversations with chat history
- Document retrieval from a vectorized knowledge base
- Local embeddings — no API needed for document indexing
- Support for PDF and TXT documents
- Deployed with Streamlit UI

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the local embedding model** (one-time):
   ```bash
   python saving_embedding_model.py
   ```
   This creates the `local_bge_model/` folder. Embeddings run entirely offline after this.

3. **Create a `.env` file** with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
   Get your API key from [Google AI Studio](https://aistudio.google.com/).

4. **Add documents** to the `knowledge_base/` folder (PDF or TXT format).

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```

6. Open your browser to the URL shown in the terminal (usually `http://localhost:8501`).

## Project Structure

```
├── app.py                      # Streamlit UI
├── document_processor.py       # Document loading and chunking
├── vector_store.py             # ChromaDB vector store with local BGE embeddings
├── rag_chain.py                # RAG chain with Gemini 2.5 Flash
├── saving_embedding_model.py   # One-time script to download the embedding model
├── knowledge_base/             # Your documents here
├── local_bge_model/            # Locally saved BGE embedding model
├── chroma_db/                  # Auto-generated vector store
├── requirements.txt
└── .env
```

## How It Works

1. **Documents** are loaded and split into chunks
2. Chunks are converted to embeddings using a **local BGE model** and stored in **ChromaDB**
3. When you ask a question, the chatbot finds the most relevant chunks and sends them to **Gemini 2.5 Flash** along with your conversation history
4. Gemini generates an answer based on your documents

