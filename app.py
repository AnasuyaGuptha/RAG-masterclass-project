import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="Zyro HR Help Desk",
    page_icon="🏢"
)

st.title("🏢 Zyro Dynamics HR Help Desk")


@st.cache_resource
def load_pipeline():

    corpus_path = os.environ.get(
        "CORPUS_PATH",
        os.path.join(
            os.path.dirname(__file__),
            "zyro-dynamics-hr-corpus"
        ),
    )

    # Load PDFs
    loader = PyPDFDirectoryLoader(corpus_path)
    docs = loader.load()

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
    )

    # Vector Store
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    # API Key
    groq_key = (
        st.secrets.get("GROQ_API_KEY")
        or os.environ.get("GROQ_API_KEY")
    )

    if not groq_key:
        st.error(
            "GROQ_API_KEY not found! "
            "Add it in Streamlit Secrets."
        )
        st.stop()

    # LLM
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.1,
        max_tokens=300,
        api_key=groq_key,
    )

    # RAG Prompt
    rag_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an HR assistant for Zyro Dynamics.

Answer ONLY using the retrieved context.

RULES:

1. Use only information explicitly present in the provided context.
2. Do not use external knowledge.
3. Do not infer, assume, or invent information.

4. If the answer is not explicitly available in the context, respond exactly:

I cannot answer this based on the available HR policy documents.

5. Answer only the specific question asked.
6. Do NOT reproduce entire policy documents.
7. Keep answers concise and focused.
8. Maximum answer length: 120 words.
9. Use bullet points when appropriate.

10. Always include exact:
   - numbers
   - durations
   - dates
   - percentages
   - amounts
   - eligibility criteria
   - approval requirements

11. Mention the relevant policy name whenever available.
12. If multiple policies are retrieved, answer only from the policy relevant to the question.
13. Do not include unrelated sections.
"""
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}

Provide a concise answer focused only on the information required to answer the question.
"""
        )
    ])

    # Out-of-Scope Classifier
    oos_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a classifier for an HR help desk.

Determine if the question can be answered using Zyro Dynamics HR policy documents.

Topics covered:
- Company Profile
- Employee Handbook
- Leave Policy
- Work From Home Policy
- Code of Conduct
- Performance Review Policy
- Compensation & Benefits Policy
- Insurance
- ESOPs
- IT & Data Security Policy
- POSH
- Onboarding & Separation Policy
- Travel & Expense Policy

Respond with EXACTLY ONE WORD:

IN_SCOPE

or

OUT_OF_SCOPE
"""
        ),
        (
            "human",
            "Question: {question}"
        )
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join([
            f"[Document: {os.path.basename(d.metadata.get('source', 'Unknown'))}]\n"
            f"{d.page_content}"
            for d in docs
        ])

    return (
        retriever,
        llm,
        rag_prompt,
        oos_prompt,
        format_docs
    )


# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if "sources" in msg and msg["sources"]:

            with st.expander("Sources"):

                for source in msg["sources"]:
                    st.write(
                        f"• {os.path.basename(source)}"
                    )


# User Input
if prompt := st.chat_input("Ask your HR question..."):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Searching HR policies..."):

            (
                retriever,
                llm,
                rag_prompt,
                oos_prompt,
                format_docs
            ) = load_pipeline()

            # Out-of-Scope Check
            guard_chain = (
                oos_prompt
                | llm
                | StrOutputParser()
            )

            guard_result = guard_chain.invoke(
                {
                    "question": prompt
                }
            )

            if guard_result.strip().upper() != "IN_SCOPE":

                answer = (
                    "I can only answer questions about Zyro Dynamics HR policies "
                    "from the provided documents."
                )

                sources = []

            else:

                docs = retriever.invoke(prompt)

                context = format_docs(docs)

                rag_chain = (
                    rag_prompt
                    | llm
                    | StrOutputParser()
                )

                answer = rag_chain.invoke(
                    {
                        "context": context,
                        "question": prompt
                    }
                )

                sources = list(
                    set(
                        d.metadata.get(
                            "source",
                            "Unknown"
                        )
                        for d in docs
                    )
                )

            st.markdown(answer)

            if sources:

                with st.expander("Sources"):

                    for source in sources:
                        st.write(
                            f"• {os.path.basename(source)}"
                        )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )
