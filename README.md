---

# 🤗 [👉Hugging Face Deployed Streamlit App Link👈](https://huggingface.co/spaces/trohith89/Novel-Office-AI-Assesment)

---

# 📚 Indian Legal Conversational Chatbot – Law Q&A over 231+ Page PDF (DU LL.B)

An intelligent RAG-based chatbot that helps students and legal researchers explore **Jurisprudence–I (Legal Method)** from Delhi University. Built using **Groq API + llama3**, **LangChain**, and **ChromaDB**, it enables natural language questions on dense legal theory. [TextBook Link](https://lawfaculty.du.ac.in/userfiles/downloads/LLBCM/Ist%20Term_Jurisprudence-I_LB101_2023.pdf)

---

## 🌟 Key Capabilities

- ⚖️ **Law-Aware RAG Pipeline**: Retrieve key theories, philosophers, and case studies using vector similarity search.
- 🧠 **LLM Reasoning via Groq**: Uses llama3 LLM on Groq's ultra-fast inference engine for fast, coherent answers.
- 🧬 **Gemini Embeddings**: Generates context-rich embeddings using GoogleGenerativeAIEmbeddings.
- 💬 **Streamlit Chatbot UI**: Clean, intuitive interface for interactive legal Q&A sessions.
- 🧩 **LangChain Modular Chains**: Uses `RunnableParallel`, `RunnableLambda`, `ChatPromptTemplate`, and memory injection.

---

📚 PDF Embedding & Vector Indexing (ChromaDB)
This component loads and processes the Jurisprudence–I PDF, splits it into manageable chunks, and generates semantic vector embeddings for legal search.

🔍 Features:
- 📖 Recursive character chunking for coherent context segments

- 🧠 Embedding with GoogleGenerativeAIEmbeddings (Gemini)

- 🗂️ Persisted in ChromaDB for fast MMR-based retrieval
---


---
🤖 AI-Powered Legal Reasoning (Groq + LangChain)
The chatbot responds to queries about natural law, positivism, legal realism, and more, using a structured LangChain pipeline.

🔍 Functional Highlights
- 🔄 Uses RunnableParallel to fetch context + history simultaneously

- 📜 Memory injection with RunnableLambda and RunnablePassthrough.assign()

- 🧾 Cites chunk references (WIP for transparency)

- 🚀 Powered by Groq API (llama3) for lightning-fast responses
---

---
💬 Streamlit Chat Interface – Legal Q&A
The frontend is built using Streamlit for easy exploration of the jurisprudence textbook via chat.

🎯 Objectives
- Support follow-up questions and memory tracking

- Reference content from relevant textbook chunks

- Break down legal theories in plain English

🧩 Core Tech Stack
- Component	Technology
- Vector DB	Chroma + Gemini Embeddings
- LLM	Groq (llama3) via LangChain
- Prompting	ChatPromptTemplate + Memory
- UI	Streamlit
---


---

🧪 Sample Questions to Ask
Try these inside the chat interface:

"Explain the concept of legal positivism in simple terms."

"What does Kelsen say about the hierarchy of norms?"

"How is Natural Law different from Legal Realism?"

"Summarize the theories of H.L.A. Hart from the book."

---

🚀 Hugging Face Directory:
- app.py

- requirements.txt

- Vector Store ChromaDB folder

Set secrets via Hugging Face Spaces settings

---
💡 Enhancements
🔎 Highlighted Summarized and `Explain it like a five` Feature in the Backend

---
🧑‍⚖️ Audience
Perfect for:

DU LL.B. students

Judiciary exam aspirants

Legal researchers and analysts

Professors explaining jurisprudence fundamentals


📬 Contact
Email: trohith89@gmail.com

LinkedIn: https://www.linkedin.com/in/trohith89/

---
