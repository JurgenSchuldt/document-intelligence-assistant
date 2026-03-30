import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Document Intelligence Assistant",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .block-container { padding-top: 1.5rem; }
    .dashboard-header {
        background-color: #1e3a5f;
        padding: 16px 24px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .dashboard-title { color: white; font-size: 20px; font-weight: 600; margin: 0; }
    .dashboard-sub { color: #8ab4d4; font-size: 13px; margin: 0; }
    .source-box {
        background: white;
        border-left: 4px solid #1e3a5f;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 12px;
        color: #555;
    }
    .section-title {
        font-size: 13px; font-weight: 600;
        color: #1e3a5f; margin-bottom: 8px;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dashboard-header">
    <p class="dashboard-title">Document Intelligence Assistant</p>
    <p class="dashboard-sub">Análisis de documentos corporativos con IA generativa — RAG con Gemini</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.load_local(
        "data/vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

@st.cache_resource
def load_chain(_vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1
    )

    retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate.from_template("""Eres un asistente experto en análisis de documentos corporativos.
Responde la pregunta basándote ÚNICAMENTE en el contexto proporcionado.
Si la información no está en el contexto, di claramente que no encuentras esa información en el documento.
Responde siempre en español y de forma clara y estructurada.

Contexto:
{context}

Pregunta: {question}

Respuesta:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

col_chat, col_info = st.columns([2, 1])

with col_info:
    st.markdown('<p class="section-title">Documento cargado</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:white;border-radius:8px;padding:12px 16px;border-left:4px solid #1e3a5f;">
        <p style="font-size:13px;font-weight:600;color:#1e3a5f;margin:0">Inditex — IAR 2025</p>
        <p style="font-size:12px;color:#888;margin:4px 0 0">Informe Anual de Remuneraciones</p>
        <p style="font-size:12px;color:#888;margin:2px 0 0">51 páginas · 254 chunks</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Preguntas de ejemplo</p>', unsafe_allow_html=True)

    example_questions = [
        "¿Cuánto cobró el consejero delegado en 2025?",
        "¿Cuáles son los objetivos de sostenibilidad?",
        "¿Quiénes forman la Comisión de Retribuciones?",
        "¿Cuál es la retribución fija de cada consejero?",
        "¿Qué es el Plan de Incentivo a Largo Plazo?"
    ]

    for q in example_questions:
        if st.button(q, key=q):
            st.session_state.pregunta_ejemplo = q

with col_chat:
    st.markdown('<p class="section-title">Pregunta al documento</p>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("Ver fuentes"):
                    for i, src in enumerate(message["sources"]):
                        st.markdown(f'<div class="source-box"><b>Fuente {i+1}</b> — Página {src.metadata.get("page", "?")+1}<br>{src.page_content[:300]}...</div>', unsafe_allow_html=True)

    pregunta = st.chat_input("Escribe tu pregunta sobre el documento...")

    if "pregunta_ejemplo" in st.session_state:
        pregunta = st.session_state.pregunta_ejemplo
        del st.session_state.pregunta_ejemplo

    if pregunta:
        st.session_state.messages.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        with st.chat_message("assistant"):
            with st.spinner("Analizando el documento..."):
                try:
                    vectorstore = load_vectorstore()
                    chain, retriever = load_chain(vectorstore)

                    sources = retriever.invoke(pregunta)
                    respuesta = chain.invoke(pregunta)

                    st.markdown(respuesta)

                    with st.expander("Ver fuentes"):
                        for i, src in enumerate(sources):
                            st.markdown(f'<div class="source-box"><b>Fuente {i+1}</b> — Página {src.metadata.get("page", "?")+1}<br>{src.page_content[:300]}...</div>', unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": respuesta,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"Error: {e}")