# Context-Aware Chatbot Using RAG

A conversational chatbot that remembers context and retrieves external information using Retrieval-Augmented Generation (RAG) with LangChain, ChromaDB, and Google Gemini 2.5 Flash.

## Features

- Context-aware conversations with chat history
- Document retrieval from a vectorized knowledge base
- Support for PDF and TXT documents
- Deployed with Streamlit UI

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/).

3. Add documents to the `knowledge_base/` folder (PDF or TXT format).

4. Run the app:
```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                  # Streamlit UI
├── document_processor.py   # Document loading and chunking
├── vector_store.py         # ChromaDB vector store
├── rag_chain.py            # RAG chain with Gemini
├── knowledge_base/         # Your documents here
├── chroma_db/              # Auto-generated vector store
├── requirements.txt
└── .env
```
