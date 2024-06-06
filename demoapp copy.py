import os
import PyPDF2
import streamlit as st
from io import StringIO
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

st.set_page_config(page_title="Demo", page_icon=':book:')

@st.cache_data
def load_docs(files):
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Por favor, proporcione un archivo txt o pdf.', icon="⚠️")
    return all_text

@st.cache_resource
def create_retriever(_embeddings, splits, retriever_type):
    if retriever_type == "BÚSQUEDA DE SIMILITUD":
        try:
            vectorstore = FAISS.from_texts(splits, _embeddings)
        except (IndexError, ValueError) as e:
            st.error(f"Error al crear el vectorstore: {e}")
            return
        retriever = vectorstore.as_retriever(k=5)
    return retriever

@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):
    st.info("`Dividiendo documento ...`")
    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Error al dividir el documento")
        st.stop()
    return splits

def main():
    st.write(
        f"""
        <div style="display: flex; align-items: center; margin-left: 0;">
            <h1 style="display: inline-block;">RAG PITEC</h1>
            <sup style="margin-left:5px;font-size:small; color: green;">Demo Version</sup>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    logo_path = "logo.png"
    st.sidebar.image(logo_path, use_column_width=True)
    st.sidebar.title("Menú")
    model_option = st.sidebar.selectbox("Selecciona el modelo de lenguaje:", ["gpt-3.5-turbo", "gpt-4o"])

    splitter_type = "RecursiveCharacterTextSplitter"
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    uploaded_files = st.file_uploader("Sube un documento PDF o TXT", type=["pdf", "txt"], accept_multiple_files=True)
    if uploaded_files:
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files

        loaded_text = load_docs(uploaded_files)
        st.write("Documentos cargados y procesados.")

        splits = split_texts(loaded_text, chunk_size=1000, overlap=0, split_method=splitter_type)

        embeddings = OpenAIEmbeddings()
        retriever = create_retriever(embeddings, splits, "BÚSQUEDA DE SIMILITUD")

        callback_handler = StreamingStdOutCallbackHandler()
        chat_openai = ChatOpenAI(model=model_option, streaming=True, verbose=True, temperature=0.8)
        qa = RetrievalQA.from_chain_type(llm=chat_openai, retriever=retriever, chain_type="stuff", verbose=True)

        predefined_questions = [
            "¿Cuál es el propósito principal del documento?",
            "¿Cuál es el resumen general del documento?",
            "¿Quiénes son las partes involucradas en este documento?",
            "¿Cuál es el rol de cada parte en este documento?",
            "¿Cuáles son los puntos clave mencionados en el documento?",
            "¿Qué términos legales importantes se utilizan en el documento y qué significan en este contexto?",
            "¿Cuáles son los acuerdos principales establecidos en el documento?",
            "¿Qué condiciones específicas deben cumplirse según el documento?",
            "¿Qué obligaciones tienen las partes involucradas según el documento?",
            "¿Cuáles son las responsabilidades asignadas a cada parte?",
            "¿Cuáles son los plazos y fechas importantes mencionados en el documento?",
            "¿Qué eventos o acciones están programados para estas fechas?",
            "¿Cuáles son las consecuencias de no cumplir con las condiciones del documento?",
            "¿Qué penalidades se mencionan en caso de incumplimiento?",
            "¿Qué leyes o normativas se mencionan o se aplican en el documento?",
            "¿Cómo se relacionan estas referencias legales con el contenido del documento?",
            "¿Existen excepciones o cláusulas especiales mencionadas en el documento?",
            "¿Cuáles son las implicaciones de estas excepciones o cláusulas?",
            "¿Qué acciones se recomiendan según el documento?",
            "¿Cuáles son los siguientes pasos sugeridos para las partes involucradas?"
        ]

        if 'predefined_answers' not in st.session_state:
            st.session_state.predefined_answers = {}
            for question in predefined_questions:
                st.sidebar.markdown(f"### {question}")
                answer = qa.run(question)
                st.session_state.predefined_answers[question] = answer
                st.sidebar.markdown(f"**Respuesta:** {answer}")
                st.sidebar.markdown("---")  # Separation line
        else:
            for question in predefined_questions:
                st.sidebar.markdown(f"### {question}")
                st.sidebar.markdown(f"**Respuesta:** {st.session_state.predefined_answers[question]}")
                st.sidebar.markdown("---")  # Separation line

        user_question = st.text_input("Ingresa tu pregunta:")

        if user_question:
            answer = qa.run(user_question)
            st.write("Respuesta:", answer)

if __name__ == "__main__":
    main()
