import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="\U0001f3e2")
st.title("\U0001f3e2 Zyro Dynamics HR Help Desk")

@st.cache_resource
def load_pipeline():
    corpus_path = os.environ.get(
        "CORPUS_PATH",
        os.path.join(os.path.dirname(__file__), "zyro-dynamics-hr-corpus"),
    )

    loader = PyPDFDirectoryLoader(corpus_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )

    groq_key = (
        st.secrets.get("GROQ_API_KEY")
        or os.environ.get("GROQ_API_KEY")
    )
    if not groq_key:
        st.error("GROQ_API_KEY not found! Set it in Streamlit Cloud → Settings → Secrets as: GROQ_API_KEY = \"your-key-here\"")
        st.stop()

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.1,
        max_tokens=512,
        api_key=groq_key,
    )

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an HR assistant for Zyro Dynamics (also referred to as Acrux Dynamics).
Answer using ONLY the provided context.

CRITICAL RULES:
- Extract exact numbers, days, months, percentages, and amounts from the context.
- When asked about timelines, cite the EXACT duration and condition from policy.
- Differentiate clearly between different leave types, insurance types, and policy sections.
- If context mentions multiple similar items, answer ONLY about the specific one asked.
- The context IS sufficient if it contains the policy rules that answer the question.
- Cite the source policy name in your answer.
- If the context lacks information, say: "I cannot answer this based on the available HR policy documents."
- Be concise and accurate."""),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    oos_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a classifier for an HR help desk.
Determine if the question can be answered using Zyro Dynamics HR policy documents.
Topics covered: company profile, employee handbook, leave policy (sick, casual, earned, maternity),
work from home, code of conduct, performance review, compensation & benefits (salary, insurance, ESOPs),
IT & data security, POSH, onboarding & separation, travel & expense.

Respond with EXACTLY ONE WORD: "IN_SCOPE" or "OUT_OF_SCOPE".

Examples:
Q: How many sick leaves do I get? -> IN_SCOPE
Q: What is the vesting schedule for ESOP? -> IN_SCOPE
Q: What is the meaning of life? -> OUT_OF_SCOPE
Q: How do I apply for WFH? -> IN_SCOPE
Q: Tell me a joke -> OUT_OF_SCOPE
Q: What is Python programming? -> OUT_OF_SCOPE
Q: How is the claim process for medical insurance? -> IN_SCOPE
Q: What is the weather today? -> OUT_OF_SCOPE"""),
        ("human", "Question: {question}"),
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join([
            f"Source: {d.metadata.get('source', 'Unknown')}\n{d.page_content}"
            for d in docs
        ])

    return retriever, llm, rag_prompt, oos_prompt, format_docs


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"- \u2022 {s.split('/')[-1]}")

if prompt := st.chat_input("Ask your HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            retriever, llm, rag_prompt, oos_prompt, format_docs = load_pipeline()

            guard_chain = oos_prompt | llm | StrOutputParser()
            guard_result = guard_chain.invoke({"question": prompt})

            if guard_result.strip().upper() != "IN_SCOPE":
                answer = "I can only answer questions about Zyro Dynamics HR policies from the provided documents."
                sources = []
            else:
                docs = retriever.invoke(prompt)
                context = format_docs(docs)
                chain = rag_prompt | llm | StrOutputParser()
                answer = chain.invoke({"context": context, "question": prompt})
                sources = list(set(
                    d.metadata.get("source", "Unknown") for d in docs
                ))

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.write(f"- \u2022 {s.split('/')[-1]}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
