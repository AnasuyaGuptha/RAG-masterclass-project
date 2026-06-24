import os
import glob
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq
from langsmith import traceable

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered"
)

# --------------------------------------------------
# LANGSMITH
# --------------------------------------------------

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

# --------------------------------------------------
# PROMPTS
# --------------------------------------------------

REFUSAL_MESSAGE = (
    "I am sorry, I can only answer HR-related questions based on Zyro Dynamics policy documents."
)

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an HR policy assistant for Zyro Dynamics.

Rules:
1. Use ONLY the supplied context.
2. Give direct answers.
3. Include exact numbers, durations, limits and policy details.
4. Do NOT mention sources.
5. Do NOT make assumptions.

If the answer is not available in the context say:

"I could not find this information in the provided HR policy documents."

Context:
{context}

Question:
{question}

Answer:
""")

OOS_PROMPT = ChatPromptTemplate.from_template("""
You are a classifier.

Determine whether the question is related to:

- HR policies
- Employee benefits
- Leave
- Payroll
- Attendance
- Compensation
- Reimbursements
- Work from home
- Company procedures

Respond ONLY:

yes

or

no

Question:
{question}
""")

# --------------------------------------------------
# BUILD PIPELINE
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading HR documents...")
def build_pipeline():

    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

    base_dir = os.path.dirname(__file__)

    pdf_folder = os.path.join(
        base_dir,
        "zyro-dynamics-hr-corpus"
    )

    pdf_files = glob.glob(
        os.path.join(pdf_folder, "*.pdf")
    )

    if len(pdf_files) == 0:
        raise Exception(
            f"No PDF files found in {pdf_folder}"
        )

    documents = []

    for pdf in sorted(pdf_files):
        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 20
        }
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=512
    )

    return retriever, llm

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

# --------------------------------------------------
# RAG
# --------------------------------------------------

@traceable(name="rag_chain")
def rag_chain(question, retriever, llm):

    docs = retriever.invoke(question)

    context = format_docs(docs)

    chain = (
        RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": answer,
        "sources": []
    }

# --------------------------------------------------
# GUARDRAIL
# --------------------------------------------------

@traceable(name="ask_bot")
def ask_bot(question, retriever, llm):

    check_prompt = OOS_PROMPT.format(
        question=question
    )

    result = llm.invoke(
        check_prompt
    ).content.lower()

    if "yes" in result:
        return rag_chain(
            question,
            retriever,
            llm
        )

    return {
        "answer": REFUSAL_MESSAGE,
        "sources": []
    }

# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🏢 Zyro Dynamics HR Help Desk")

st.caption(
    "Ask questions about HR policies, benefits and company procedures."
)

retriever, llm = build_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input(
    "Ask an HR question..."
):

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching HR policies..."
        ):

            result = ask_bot(
                prompt,
                retriever,
                llm
            )

        st.markdown(
            result["answer"]
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })
