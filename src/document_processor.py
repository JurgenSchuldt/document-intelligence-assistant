import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def process_documents(pdf_paths, persist_directory="data/vectorstore"):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("No se encontró GOOGLE_API_KEY en el archivo .env")

    print("Cargando documentos...")
    documents = []

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        documents.extend(docs)
        print(f"  {path}: {len(docs)} páginas")

    print(f"\nTotal páginas cargadas: {len(documents)}")

    print("Dividiendo en chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks: {len(chunks)}")

    print("Generando embeddings por lotes...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    batch_size = 20
    pause_seconds = 40
    vectorstore = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Procesando lote {i // batch_size + 1} de {(len(chunks) + batch_size - 1) // batch_size}...")

        try:
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
        except Exception as e:
            print(f"Error en el lote {i // batch_size + 1}: {e}")
            print("Esperando 45 segundos antes de reintentar...")
            time.sleep(45)

            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)

        if i + batch_size < len(chunks):
            print(f"Esperando {pause_seconds} segundos para evitar exceder cuota...")
            time.sleep(pause_seconds)

    os.makedirs(persist_directory, exist_ok=True)
    vectorstore.save_local(persist_directory)
    print(f"Vectorstore guardado en {persist_directory}")

    return vectorstore

if __name__ == "__main__":
    pdfs = [f"data/{f}" for f in os.listdir("data") if f.endswith(".pdf")]

    if not pdfs:
        print("No hay PDFs en la carpeta data/")
    else:
        process_documents(pdfs)