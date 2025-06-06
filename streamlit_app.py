import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- API Keys ---
if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GROQ_API_KEY"):
    st.error("❌ Set GOOGLE_API_KEY and GROQ_API_KEY in your environment.")
    st.stop()

# --- Page Config ---
st.set_page_config(page_title="Jurisprudence Chatbot", layout="wide")


# Title
st.markdown("""<h1 style='width:100%;background-color:transparent;'>📚 Jurisprudence Conversational Chatbot</h1>""", unsafe_allow_html=True)



# --- In-Memory Chat History (Thread-safe)
if "memory_dict" not in st.session_state:
    st.session_state.memory_dict = {"history": []}

memory_dict = st.session_state.memory_dict

# --- New Chat Button ---
if st.button("🆕 New Chat"):
    st.session_state.memory_dict = {"history": []}
    st.rerun()

# --- Display Chat History like ChatGPT ---
for msg in memory_dict["history"]:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# --- Chat Input ---
question = st.chat_input("💬 Ask a question from the Jurisprudence textbook...")

if question:
    # --- Show User Message ---
    memory_dict["history"].append(HumanMessage(content=question))
    with st.chat_message("user"):
        st.markdown(question)

    # --- Embeddings + Vector DB ---
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    # --- Format Context from Docs ---
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_context(question):
        docs = retriever.get_relevant_documents(question)
        return {
            "question": question,
            "context": format_docs(docs),
            "docs": docs
        }

    # --- Prompt & LLM ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a legal assistant specialized in Jurisprudence, helping students understand complex legal topics from the textbook "Jurisprudence–I (Legal Method)".
Instructions:
- Use ONLY the provided context to answer questions.
- If the user asks for a summary, generate a concise summary of the most relevant sections.
- If the user says "Explain like I’m five" or "ELI5", simplify the legal explanation as much as possible using analogies and plain language.
- Always provide citations in the format: [Section Title or Page Number].
- If you don’t know the answer, say you don’t know instead of making it up.
Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    llm = ChatGroq(model_name="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))
    parser = StrOutputParser()

    # --- Chain with memory injected manually (no st.session_state in threads) ---
    def chain_with_memory(input_dict):
        context_info = get_context(input_dict["input"])
        chain_input = {
            "question": context_info["question"],
            "context": context_info["context"],
            "chat_history": memory_dict["history"],
        }
        answer = prompt | llm | parser
        result = answer.invoke(chain_input)
        return {"result": result, "docs": context_info["docs"]}

    chain = RunnablePassthrough.assign() | RunnableLambda(chain_with_memory)

    # --- Get AI Response ---
    output = chain.invoke({"input": question})
    response = output["result"]

    memory_dict["history"].append(AIMessage(content=response))

    # --- Show AI Response ---
    with st.chat_message("assistant"):
        st.markdown(response)
