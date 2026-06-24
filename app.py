import os
import glob
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langsmith import traceable

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered"
)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "zyro-rag-challenge"

REFUSAL_MESSAGE = "I am sorry, I can only answer HR-related questions based on Zyro Dynamics policy documents. Please contact your HR team for other queries."

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an HR policy assistant.

Use ONLY the supplied context.

Rules:
1. Answer directly.
2. Include exact numbers, dates, durations, percentages, grades, and policy names whenever available.
3. If multiple policy clauses apply, combine them.
4. Do not make assumptions.
5. If the answer is not in the context, say:
   "I could not find this information in the provided HR policy documents."

Context:
{context}

Question:
{question}

Answer:
""")

@st.cache_resource(show_spinner="Loading HR documents and building knowledge base...")
def build_pipeline():
    os.environ["GROQ_API_KEY"]      = st.secrets["GROQ_API_KEY"]
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

    pdf_files = glob.glob("hr_docs/*.pdf")
    documents = []
    for pdf_path in sorted(pdf_files):
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    chunks   = splitter.split_documents(documents)

    embeddings  = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20}
    )

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=512)
    return retriever, llm

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}]\n{doc.page_content}"
        for doc in docs
    )

@traceable(name="rag_chain")
def rag_chain(question, retriever, llm):
    docs     = retriever.invoke(question)
    context  = format_docs(docs)
    prompt   = RAG_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)
    sources  = list({os.path.basename(doc.metadata.get("source", "Unknown")) for doc in docs})
    return {"answer": response.content, "sources": sources}

@traceable(name="ask_bot")
def ask_bot(question, retriever, llm):
    check    = OOS_PROMPT.format(question=question)
    response = llm.invoke(check)
    if "yes" in response.content.strip().lower():
        return rag_chain(question, retriever, llm)
    return {"answer": REFUSAL_MESSAGE, "sources": []}

# ── UI ──────────────────────────────────────────────────────
st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask any HR policy question — powered by RAG")

retriever, llm = build_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.caption(s)

if question := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            result = ask_bot(question, retriever, llm)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("📄 Sources"):
                for s in result["sources"]:
                    st.caption(s)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })

    
