# Context-Aware Chatbot Using RAG — Developer Guide

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Core Concepts Explained](#core-concepts-explained)
   - [What is an LLM?](#what-is-an-llm)
   - [What is RAG?](#what-is-rag)
   - [What are Embeddings?](#what-are-embeddings)
   - [What is a Vector Database?](#what-is-a-vector-database)
   - [What is LangChain?](#what-is-langchain)
   - [What is Context Memory?](#what-is-context-memory)
3. [How the Pieces Fit Together](#how-the-pieces-fit-together)
4. [Project Structure](#project-structure)
5. [File-by-File Breakdown](#file-by-file-breakdown)
6. [Data Flow: From Question to Answer](#data-flow-from-question-to-answer)
7. [Setup and Installation](#setup-and-installation)
8. [How to Add Your Own Documents](#how-to-add-your-own-documents)
9. [How to Customize](#how-to-customize)
10. [Troubleshooting](#troubleshooting)

---

## What This Project Does

This project builds a chatbot that:

1. **Reads your documents** (PDFs, text files) — like a company handbook, product manual, or any knowledge base.
2. **Remembers conversation context** — if you say "What about the second point?" it knows what you're referring to.
3. **Answers questions using your documents** — instead of making things up, it finds relevant information from your files and uses that to respond.

Think of it like giving a smart assistant a library of documents to reference before answering your questions.

---

## Core Concepts Explained

### What is an LLM?

**LLM = Large Language Model**. It's a type of AI that can understand and generate human-like text. Examples: GPT-4, Gemini, Claude.

- You give it a prompt (text input), it gives you a response (text output).
- It's trained on massive amounts of internet text, so it knows a lot — but it **doesn't know your private documents** unless you show them to it.
- This project uses **Google Gemini** as the LLM.

```python
# Simple example of using an LLM (conceptual):
response = llm.invoke("What is Python?")
# Returns: "Python is a high-level programming language..."
```

### What is RAG?

**RAG = Retrieval-Augmented Generation**. It's a technique that combines two steps:

1. **Retrieval**: Find relevant documents from your knowledge base.
2. **Generation**: Use an LLM to generate an answer based on those documents.

**Why RAG?** Without RAG, an LLM only knows what it was trained on. If you ask about your company's internal policies, it won't know. RAG solves this by:
- Storing your documents in a searchable format
- Finding the relevant pieces when a question is asked
- Feeding those pieces to the LLM along with the question

**Analogy**: Imagine taking an open-book exam. Instead of memorizing everything, you look up the relevant chapter (retrieval) and then write your answer (generation). That's RAG.

### What are Embeddings?

**Embeddings are numerical representations of text.** Computers can't understand words directly, so we convert text into lists of numbers (vectors).

- The sentence "I love dogs" might become: `[0.12, -0.45, 0.78, ...]` (hundreds of numbers)
- Similar meanings get similar numbers. "I love dogs" and "I like canines" will have close numerical values.
- "I love dogs" and "The stock market crashed" will have very different numerical values.

```python
# Conceptual example:
embedding("I love dogs")       → [0.12, -0.45, 0.78, ...]
embedding("I like canines")    → [0.11, -0.43, 0.76, ...]  ← similar!
embedding("Stock market crash") → [0.89, 0.34, -0.12, ...]  ← very different
```

We use **Google's embedding model** to convert text into these number vectors.

### What is a Vector Database?

A **vector database** stores embeddings (those number lists) and lets you search by meaning, not by exact keyword match.

**Traditional database search** (keyword match):
- You search for "canine" → finds documents containing the word "canine"
- Misses documents that say "dog" even though they mean the same thing

**Vector database search** (semantic/meaning match):
- You search for "canine" → finds documents about dogs, even if they never use the word "canine"
- Works because "canine" and "dog" have similar embeddings

This project uses **ChromaDB** — a lightweight vector database that stores data on your local machine.

```python
# Conceptual example:
vector_store.add("Dogs are loyal pets")
vector_store.add("Cats are independent animals")

results = vector_store.search("What makes good companions?")
# Returns: "Dogs are loyal pets" (because it's semantically closest)
```

### What is LangChain?

**LangChain is a Python library that makes it easier to build LLM applications.** Think of it as a toolkit that provides:

- **Document loaders**: Read PDFs, text files, websites, etc.
- **Text splitters**: Break large documents into smaller chunks
- **Vector store integrations**: Connect to ChromaDB and other databases
- **Chain builders**: Connect retrieval + LLM into a single pipeline
- **Memory management**: Track conversation history

Without LangChain, you'd write hundreds of lines of glue code. With it, you compose building blocks.

```python
# Without LangChain: you'd manually handle embeddings, vector search, prompt building, API calls...
# With LangChain:
chain = create_retrieval_chain(retriever, question_answer_chain)
result = chain.invoke({"input": "What are dogs?"})
```

### What is Context Memory?

**Context memory** means the chatbot remembers what was said earlier in the conversation.

**Without memory:**
```
User: What is Python?
Bot: Python is a programming language...
User: What about its second feature?
Bot: I don't know what "second feature" you mean. (no memory!)
```

**With memory:**
```
User: What is Python?
Bot: Python is a programming language...
User: What about its second feature?
Bot: Python's second key feature is its extensive standard library... (remembers the context!)
```

This project implements context memory by sending the full conversation history to the LLM with each new question, so it understands references like "it", "that", "the second point", etc.

---

## How the Pieces Fit Together

```
┌─────────────────────────────────────────────────────────────┐
│                        YOUR DOCUMENTS                        │
│                   (PDFs, TXT files in folder)                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    document_processor.py
                    (loads & splits text)
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    CHUNKS OF TEXT                            │
│              (small pieces of your documents)                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                     vector_store.py
                  (converts to embeddings
                   and stores in ChromaDB)
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     ChromaDB                                 │
│              (vector database on your disk)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                      rag_chain.py
              (finds relevant chunks + asks Gemini)
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Gemini 2.0 Flash                          │
│                   (the AI brain)                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                          app.py
                    (Streamlit web UI)
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      USER                                    │
│              (asks questions in browser)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
context-aware-chatbot-using-RAG/
│
├── app.py                    # Streamlit web interface (the UI)
├── document_processor.py     # Loads documents and splits them into chunks
├── vector_store.py           # Creates/loads the ChromaDB vector database
├── rag_chain.py              # Builds the RAG pipeline (retrieval + LLM)
├── requirements.txt          # Python dependencies
├── README.md                 # Quick start guide
├── DOCS.md                   # This file — detailed developer guide
│
├── knowledge_base/           # Put your PDF/TXT documents here
│   └── machine_learning_intro.txt   # Sample document included
│
├── chroma_db/                # Auto-generated vector database (created on first run)
│
└── .env                      # Your Google API key (create this yourself)
```

---

## File-by-File Breakdown

### `document_processor.py`

**Purpose**: Read your documents and break them into manageable pieces.

**Two functions:**

1. **`load_documents(directory)`** — Scans a folder for `.pdf` and `.txt` files and reads their contents.
   - Uses `PyPDFLoader` for PDFs (extracts text from each page)
   - Uses `TextLoader` for plain text files
   - Returns a list of Document objects (each has `.page_content` and `.metadata`)

2. **`chunk_documents(documents)`** — Splits large documents into smaller pieces.
   - Why chunk? You can't fit a 200-page document into a single LLM prompt. We break it into ~1000-character pieces.
   - Uses `RecursiveCharacterTextSplitter` which splits on natural boundaries (paragraphs → sentences → words)
   - `chunk_overlap=200` means adjacent chunks share 200 characters of overlap, so context isn't lost at boundaries

```python
# What happens internally:
# A 5000-word document becomes ~50 chunks of ~1000 characters each
# Each chunk is stored separately so we can retrieve only the relevant ones
```

### `vector_store.py`

**Purpose**: Convert text chunks into embeddings and store them in ChromaDB.

**Two functions:**

1. **`create_vector_store(documents)`** — Takes document chunks, converts each to an embedding using Google's embedding model, and saves them to ChromaDB on disk.

2. **`load_vector_store()`** — Loads an existing ChromaDB database from disk (so you don't need to re-process documents every time).

```python
# What happens internally:
# "Dogs are loyal" → embedding model → [0.12, -0.45, 0.78, ...] → stored in ChromaDB
# "Cats are independent" → embedding model → [0.55, 0.22, -0.33, ...] → stored in ChromaDB
# When you search "good pets", ChromaDB finds the closest embedding match
```

### `rag_chain.py`

**Purpose**: Build the complete RAG pipeline — the brain of the chatbot.

**What it does step by step:**

1. **Creates the LLM** — Initializes Google Gemini 2.0 Flash as the language model.

2. **Creates the retriever** — Wraps the vector store so it can find the top 3 most relevant document chunks for any query.

3. **Creates a context-aware retriever** — This is the clever part. Before searching, it reformulates the user's question using conversation history. If you say "tell me more about that", it figures out what "that" refers to before searching.

4. **Creates the question-answer chain** — Builds a prompt template that combines:
   - System instructions ("You are a helpful assistant...")
   - Retrieved document context
   - Conversation history
   - The user's question

5. **Combines everything** into a single `rag_chain` that you can call with `chain.invoke({"input": "question", "chat_history": [...]})`.

```python
# The prompt that goes to Gemini looks like this:
#
# System: You are a helpful AI assistant. Use the following retrieved context
#         to answer the user's question...
#
# Context: [Chunk 1 from your documents]
#          [Chunk 2 from your documents]
#          [Chunk 3 from your documents]
#
# Chat History:
#   User: What is Python?
#   Assistant: Python is a programming language...
#
# Human: What about its second feature?
```

### `app.py`

**Purpose**: The web interface built with Streamlit.

**What it does:**

1. **Initializes the chatbot** on first load — either loads an existing vector store or processes documents and creates one.

2. **Displays chat history** — Shows all previous messages in a chat-like format.

3. **Handles user input** — When you type a question:
   - Adds it to chat history
   - Sends it to the RAG chain (with conversation history)
   - Displays the response

4. **Sidebar controls**:
   - "Clear Chat" — Resets conversation history
   - "Rebuild Vector Store" — Deletes and recreates the database (useful after adding new documents)

**Streamlit basics**: Streamlit is a Python library that turns scripts into web apps. `st.chat_message()` creates speech bubbles, `st.chat_input()` creates a text input box, and the page re-renders every time you interact with it.

---

## Data Flow: From Question to Answer

Here's exactly what happens when a user asks a question:

```
Step 1: User types "What are the types of machine learning?"
        ↓
Step 2: app.py adds this to chat_history
        ↓
Step 3: rag_chain receives the question + chat_history
        ↓
Step 4: The context-aware retriever reformulates the question
        (if needed, using chat history for context)
        ↓
Step 5: The reformulated question is converted to an embedding
        ↓
Step 6: ChromaDB searches for the 3 most similar document chunks
        ↓
Step 7: These chunks are inserted into the prompt as "context"
        ↓
Step 8: The full prompt (system + context + history + question) is sent to Gemini
        ↓
Step 9: Gemini generates an answer based on the provided context
        ↓
Step 10: The answer is displayed to the user and saved to chat_history
```

---

## Setup and Installation

### Prerequisites

- **Python 3.10+** installed
- **Google API key** from [Google AI Studio](https://aistudio.google.com/) (free tier available)

### Steps

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create a `.env` file** in the project root:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

3. **Add documents** to the `knowledge_base/` folder:
   - Supported formats: `.pdf`, `.txt`
   - Add as many as you want

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to the URL shown in the terminal (usually `http://localhost:8501`)

---

## How to Add Your Own Documents

1. Place `.pdf` or `.txt` files in the `knowledge_base/` folder.

2. In the Streamlit UI, click **"Rebuild Vector Store"** in the sidebar (or delete the `chroma_db/` folder and restart the app).

3. Start asking questions about your documents!

**Tips:**
- Text files work best (PDFs can have formatting issues)
- Documents should be in a language the embedding model supports (English works best)
- Larger documents are automatically split into chunks — you don't need to split them manually

---

## How to Customize

### Change the LLM model

In `rag_chain.py`, modify the model name:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",   # Change to another Gemini model
    ...
)
```

### Change chunk size

In `document_processor.py`, adjust the parameters:

```python
def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    # Larger chunks = more context per chunk, but slower search
    # Smaller chunks = more precise retrieval, but may lose context
```

### Change number of retrieved documents

In `rag_chain.py`, modify `search_kwargs`:

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
# Change k=3 to retrieve more or fewer chunks
```

### Change the system prompt

In `rag_chain.py`, edit the `qa_system_prompt`:

```python
qa_system_prompt = (
    "You are a helpful AI assistant. Use the following retrieved context "
    "to answer the user's question. If you don't know the answer, say so. "
    "Keep answers concise and helpful.\n\n"
    "{context}"
)
# Modify the instructions to change the bot's behavior
```

### Use a different embedding model

In `vector_store.py`, change the embedding model:

```python
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# Other options: "models/text-embedding-004"
```

---

## Troubleshooting

### "GOOGLE_API_KEY not found"
- Make sure you created a `.env` file in the project root
- The file should contain exactly: `GOOGLE_API_KEY=your_key_here`
- Restart the app after creating/editing `.env`

### "No documents found"
- Check that files exist in the `knowledge_base/` folder
- Only `.pdf` and `.txt` files are supported
- Click "Rebuild Vector Store" in the sidebar

### Slow responses
- Large documents take longer to process initially (one-time cost)
- After the vector store is built, responses should be fast
- Try reducing `chunk_size` or `k` (number of retrieved chunks)

### Incorrect answers
- Make sure your documents actually contain the information being asked about
- Try increasing `k` in `rag_chain.py` to retrieve more context
- Check that documents are readable (PDFs with images/scans won't work — they need selectable text)

### Import errors
- Run `pip install -r requirements.txt` to ensure all packages are installed
- Make sure you're using Python 3.10 or higher
