
```markdown
# Document Intelligence Assistant

Asistente de IA para análisis de documentos corporativos usando RAG con Gemini.

Live demo: https://document-intelligence-assistant.streamlit.app/

## Qué hace

Permite hacer preguntas en lenguaje natural sobre documentos PDF corporativos. El sistema busca los fragmentos más relevantes del documento y genera respuestas precisas basándose únicamente en el contenido del PDF, sin inventarse información.

Documento de prueba: Informe Anual de Remuneraciones de Inditex 2025 (51 páginas).

## Ejemplos de preguntas

- ¿Cuánto cobró el consejero delegado en 2025?
- ¿Cuáles son los objetivos de sostenibilidad?
- ¿Quiénes forman la Comisión de Retribuciones?
- ¿Qué es el Plan de Incentivo a Largo Plazo?

## Arquitectura

El sistema usa RAG — Retrieval Augmented Generation:

1. El PDF se divide en 254 chunks de texto
2. Cada chunk se convierte en embeddings con Gemini Embedding
3. Los embeddings se guardan en un vectorstore FAISS
4. Cuando el usuario hace una pregunta, el sistema recupera los 4 chunks más relevantes
5. Gemini genera una respuesta basándose únicamente en esos chunks

## Tecnologías

- Python 3.11
- LangChain
- Google Gemini API
- FAISS
- Streamlit
- PyPDF

## Cómo ejecutarlo

```bash
git clone https://github.com/JurgenSchuldt/document-intelligence-assistant.git
cd document-intelligence-assistant
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Añade tu GOOGLE_API_KEY en un archivo `.env`:


GOOGLE_API_KEY=tu-key-aqui
