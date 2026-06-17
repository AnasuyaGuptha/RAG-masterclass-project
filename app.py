
import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_groq import ChatGroq

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

CORPUS_PATH = "."

# Set your Groq API key as environment variable
# For Streamlit Cloud use st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1024
)

# --------------------------------------------------
# PROMPTS
# --------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_template(
'''
You are Zyro Dynamics HR Assistant.

Use ONLY the information in the provided context.

If the answer is not present in the context, respond exactly:

"I could not find this information in the Zyro Dynamics HR policy documents."

Context:
{context}

Question:
{question}

Answer:
'''
)

# --------------------------------------------------
# LOAD AND INDEX DOCUMENTS
# --------------------------------------------------

@st.cache_resource
def build_rag():

    docs = []

    for file in os.listdir(CORPUS_PATH):

        if file.endswith(".pdf"):

            loader = PyPDFLoader(
                os.path.join(CORPUS_PATH, file)
            )

            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    return retriever

retriever = build_rag()

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def format_docs(docs):

    return "\n\n".join(
        doc.page_content for doc in docs
    )

def ask_bot(question):

    docs = retriever.invoke(question)

    chain = (
        {
            "context": lambda _: format_docs(docs),
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)

    return answer, docs

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢"
)

st.title("🏢 Zyro Dynamics HR Help Desk")

st.markdown(
    "Ask questions about company policies, leave, benefits, WFH, onboarding, performance reviews and more."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Ask an HR question..."
)

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching HR policies..."):

        answer, docs = ask_bot(prompt)

    with st.chat_message("assistant"):

        st.markdown(answer)

        with st.expander("Sources"):

            for i, doc in enumerate(docs[:3], start=1):

                st.markdown(
                    f"### Source {i}"
                )

                st.write(
                    doc.page_content[:700]
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
