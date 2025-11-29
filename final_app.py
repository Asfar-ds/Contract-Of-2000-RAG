import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
groq_api = os.getenv('GROQ_API_KEY')
os.environ['GROQ_API_KEY'] = groq_api

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title='Llama Models for Contracts', page_icon=':robot_face:', layout='wide')

# Sidebar
st.sidebar.title('🏛️💬 AI Legal Assistant')
st.sidebar.write(
    '''The model is trained on 512 contract files, including Real Property Leases, Intellectual Property Assignments,
Indebtedness Agreements, Subcontracts, and Transition Services Agreements. Chat with the AI Assistant to explore these contracts.'''
)

uploaded_file = st.sidebar.file_uploader('Upload your data', type=['txt', 'pdf'])

if groq_api:
    st.sidebar.success('API Key provided ✅')
else:
    st.sidebar.error('API Key not found 🚨')

st.sidebar.subheader('Models & Parameters 🛠️')
selected_model = st.sidebar.selectbox('Choose the Model', ["Llama 3 70b", 'Llama 3 8b', 'Llama 3.1 8b instruct'])

model_mapping = {
    "Llama 3 70b": "llama3-70b-8192",
    "Llama 3 8b": "llama-3.1-8b-instant",
    "Llama 3.1 8b instruct": "llama-3.1-8b-instant"
}
model = model_mapping[selected_model]

temperature = st.sidebar.slider('Temperature', min_value=0.01, max_value=1.0, value=0.5, step=0.1)
p_value = st.sidebar.slider('Top-p', min_value=0.01, max_value=1.0, value=0.9, step=0.1)

bot_emoji = '🤖'
user_emoji = '🙋‍♂️'

# -----------------------------
# Function: Load and Split File
# -----------------------------
def load_and_split_file(file):
    if file is None:
        return None
    temp_file_path = os.path.join("temp", file.name)
    os.makedirs("temp", exist_ok=True)
    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())

    if file.type == 'text/plain':
        loader = TextLoader(temp_file_path)
    elif file.type == 'application/pdf':
        loader = PyPDFLoader(temp_file_path)
    else:
        st.error("Unsupported file type")
        return None

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# -----------------------------
# Embeddings Setup
# -----------------------------
persist_directory = 'faissembeddings'
embedding_model_name = 'sentence-transformers/all-mpnet-base-v2'

def load_embeddings(modelHF):
    hf_model = HuggingFaceEmbeddings(model_name=modelHF)

    if os.path.exists(persist_directory):
        embeddings_instance = FAISS.load_local(
            folder_path=persist_directory,
            embeddings=hf_model,
            allow_dangerous_deserialization=True
        )
    else:
        embeddings_instance = FAISS(embedding_function=hf_model, index=None)
    return embeddings_instance

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = load_embeddings(embedding_model_name)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Add Uploaded Document
# -----------------------------
def add_doc_to_embeddings(file):
    if file is not None:
        documents = load_and_split_file(file)
        if documents:
            st.session_state.embeddings.add_documents(documents)
            st.session_state.embeddings.save_local(folder_path=persist_directory)

if uploaded_file and st.sidebar.button("Add Document to Embeddings"):
    add_doc_to_embeddings(uploaded_file)

# -----------------------------
# Display Chat History
# -----------------------------
for msg in st.session_state.get("chat_history", []):
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Sample Questions
def add_sample_questions():
    st.subheader('Sample Questions:')
    st.write('👉 How do the "364-Day" and "Five-Year" Credit Agreements differ in terms of interest rates, payment, and defaults?')
    st.write("👉 What is the effect of the covenants in these credit agreements on the borrower's operations and finances?")
    st.write('👉 What provisions are made for managing confidential information in these agreements, and what are their potential impacts?')

# -----------------------------
# User Prompt
# -----------------------------
user_prompt = st.chat_input('Ask a question about contracts')

if not user_prompt:
    add_sample_questions()

if user_prompt:
    st.chat_message('user').markdown(user_prompt)
    st.session_state.chat_history.append({'role': 'user', 'content': user_prompt})

    # Retriever
    retriever = st.session_state.embeddings.as_retriever()
    retrieved_docs = retriever.retrieve(user_prompt)  # ✅ fixed method name

    # Combine context
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Initialize ChatGroq
    chat_model = ChatGroq(
        api_key=groq_api,
        model=model,
        temperature=temperature,
        top_p=p_value
    )

    # Build messages
    messages = [
        SystemMessage(content="You are a helpful assistant specialized in contracts. Answer in the user's language if requested."),
        HumanMessage(content=f"Context: {context}\n\nQuestion: {user_prompt}")
    ]

    # Generate response
    response = chat_model(messages)

    st.chat_message('assistant').markdown(response.content)
    st.session_state.chat_history.append({'role': 'assistant', 'content': response.content})
