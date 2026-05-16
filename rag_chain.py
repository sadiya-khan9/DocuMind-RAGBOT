import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(vector_store, api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the following retrieved context to answer the user's question. If you don't know the answer, say so. Keep answers concise and helpful.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def retrieve_and_answer(inputs):
        chat_history = inputs.get("chat_history", [])
        user_input = inputs["input"]

        if chat_history:
            standalone_question = contextualize_q_chain.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })
        else:
            standalone_question = user_input

        docs = retriever.invoke(standalone_question)
        context = format_docs(docs)

        response = qa_prompt | llm | StrOutputParser()
        answer = response.invoke({
            "context": context,
            "input": user_input,
            "chat_history": chat_history,
        })
        return {"answer": answer, "context": docs}

    return retrieve_and_answer
