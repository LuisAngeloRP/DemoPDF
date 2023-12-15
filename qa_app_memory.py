import os
import PyPDF2
import random
import itertools
import streamlit as st
from io import StringIO
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SVMRetriever
from langchain.chains import QAGenerationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import CallbackManager

class ChatMemory:
    def __init__(self):
        self.history = []

    def add_to_history(self, question, answer):
        self.history.append({"question": question, "answer": answer})

    def get_history(self):
        return self.history

def get_image_path(phase):
    image_folder = "images"
    image_path = os.path.join(image_folder, f"{phase.upper()}.png")
    return image_path

st.set_page_config(page_title="TOGAF con PDFs", page_icon=':book:')

@st.cache_data
def load_docs(files):
    st.info("`Leyendo TOGAF ...`")
    all_text = ""
    pdf_reader = PyPDF2.PdfReader("togaf.pdf")
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    all_text += text
    st.info("`Leyendo documento ...`")
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

@st.cache_data
def generate_eval(text, N, chunk):
    st.info("`Generando preguntas de muestra ...`")
    n = len(text)
    starting_indices = [random.randint(0, n-chunk) for _ in range(N)]
    sub_sequences = [text[i:i+chunk] for i in starting_indices]
    chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0.2))
    eval_set = []
    for i, b in enumerate(sub_sequences):
        try:
            qa = chain.run(b)
            eval_set.append(qa)
            st.write("Creando Pregunta:", i+1)
        except:
            st.warning('Error al generar la pregunta %s.' % str(i+1), icon="⚠️")
    eval_set_full = list(itertools.chain.from_iterable(eval_set))
    return eval_set_full

# ...

def main():
    foot = f"""
    <div style="
        position: fixed;
        bottom: 0;
        left: 30%;
        right: 0;
        width: 50%;
        padding: 0px 0px;
        text-align: center;
    ">
    """
    st.markdown(foot, unsafe_allow_html=True)

    # Agregar CSS personalizado
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .css-card {
            border-radius: 0px;
            padding: 30px 10px 10px 10px;
            background-color: #f8f9fa;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 10px;
            font-family: "IBM Plex Sans", sans-serif;
        }
        .card-tag {
            border-radius: 0px;
            padding: 1px 5px 1px 5px;
            margin-bottom: 10px;
            position: absolute;
            left: 0px;
            top: 0px;
            font-size: 0.6rem;
            font-family: "IBM Plex Sans", sans-serif;
            color: white;
            background-color: green;
        }
        .css-zt5igj {left:0;}
        span.css-10trblm {margin-left:0;}
        div.css-1kyxreq {margin-top: -40px;}
        </style>
        """,
        unsafe_allow_html=True,
    )   

    st.write(
        f"""
        <div style="display: flex; align-items: center; margin-left: 0;">
            <h1 style="display: inline-block;">TOGAF PDF</h1>
            <sup style="margin-left:5px;font-size:small; color: green;">beta v0.4</sup>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Menú")

    embedding_option = st.sidebar.radio(
        "Elige Embeddings", ["OpenAI Embeddings"])

    retriever_type = st.sidebar.selectbox(
        "Elige Retriever", ["BÚSQUEDA DE SIMILITUD"])

    temperature = st.sidebar.slider(
        "Temperatura", 0.0, 1.5, 0.8, step=0.1)
    
    chunk_size = st.sidebar.slider(
        "Tamaño de Chunk (chunk_size)", 100, 2000, 1000, step=100)
    
    splitter_type = "RecursiveCharacterTextSplitter"

    if 'openai_api_key' not in st.session_state:
        openai_api_key = st.text_input(
            'Por favor, ingresa tu clave de API de OpenAI o [visita aquí](https://platform.openai.com/account/api-keys)',
            value="", placeholder="Ingresa la clave de API de OpenAI que comienza con sk-")
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            return
    else:
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

    uploaded_files = st.file_uploader("Sube un documento PDF o TXT", type=[
                                      "pdf", "txt"], accept_multiple_files=True)
    
        # Inicializar el historial de chat
    if 'chat_memory' not in st.session_state:
        st.session_state.chat_memory = ChatMemory()
    

    if uploaded_files:
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files
            if 'eval_set' in st.session_state:
                del st.session_state['eval_set']

        loaded_text = load_docs(uploaded_files)
        st.write("Documentos cargados y procesados.")

        splits = split_texts(loaded_text, chunk_size=chunk_size,
                             overlap=0, split_method=splitter_type)

        num_chunks = len(splits)
        st.write(f"Número de chunks: {num_chunks}")

        if embedding_option == "OpenAI Embeddings":
            embeddings = OpenAIEmbeddings()

        retriever = create_retriever(embeddings, splits, retriever_type)

        callback_handler = StreamingStdOutCallbackHandler()
        callback_manager = CallbackManager([callback_handler])

        chat_openai = ChatOpenAI(
            streaming=True, callback_manager=callback_manager, verbose=True, temperature=temperature)
        qa = RetrievalQA.from_chain_type(llm=chat_openai, retriever=retriever, chain_type="stuff", verbose=True)

        if 'eval_set' not in st.session_state:
            num_eval_questions = 0
            st.session_state.eval_set = generate_eval(
                loaded_text, num_eval_questions, 3000)

        for i, qa_pair in enumerate(st.session_state.eval_set):
            st.sidebar.markdown(
                f"""
                <div class="css-card">
                <span class="card-tag">Pregunta {i + 1}</span>
                    <p style="font-size: 12px;">{qa_pair['question']}</p>
                    <p style="font-size: 12px;">{qa_pair['answer']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.write("Listo para responder preguntas.")

        user_question = st.text_input("Ingresa tu pregunta:")
        if user_question:
            answer = qa.run(user_question)
            st.write("Respuesta:", answer)

            # Guardar la conversación en el historial solo si hay una respuesta válida
            if answer:
                # Inicializar el historial de chat si aún no existe
                if 'chat_memory' not in st.session_state:
                    st.session_state.chat_memory = ChatMemory()

                # Guardar la conversación en el historial
                st.session_state.chat_memory.add_to_history(user_question, answer)

            st.write("Respuesta:", answer)
            
            phase_keywords = {"A": ["fase A"], "B": ["fase B"],"C": ["fase C"],"D": ["fase D"],"E": ["fase E"],"G": ["fase G"],"H": ["fase H"],"F": ["fase F"],"P": ["fase Preliminar"],}  # Puedes extender esto según tus necesidades
            selected_phase = None
            for phase, keywords in phase_keywords.items():
                if any(keyword in answer for keyword in keywords):
                    selected_phase = phase
                    break

            # Mostrar la imagen correspondiente a la fase
            if selected_phase:
                image_path = get_image_path(selected_phase)
                st.image(image_path, caption=f"Imagen correspondiente a la Fase {selected_phase}", use_column_width=True)

            # Mostrar el historial de chat con un recuadro y título más grande
            st.markdown("## Historial de Chat", unsafe_allow_html=True)
            st.markdown(
                '<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 20px;">',
                unsafe_allow_html=True
            )
            for chat_entry in st.session_state.chat_memory.get_history():
                st.write(f"Pregunta: {chat_entry['question']}")
                st.write(f"Respuesta: {chat_entry['answer']}")
                st.write("---")
            st.markdown('</div>', unsafe_allow_html=True)

            # Obtener la fase de la respuesta
  


if __name__ == "__main__":
    main()