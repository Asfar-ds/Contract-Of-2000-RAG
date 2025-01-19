import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
groq_api = os.getenv('GROQ_API_KEY')
os.environ['GROQ_API_KEY'] = groq_api  # Set the environment variable

# Streamlit app configuration
st.set_page_config(page_title='Llama Models for Contracts', page_icon=':robot_face:', layout='wide')

# Sidebar configuration
with st.sidebar:
    st.title('🦙💬 Llama Chatbot')
    st.write('This Llama model will answer questions based on a 2000-document contract knowledge base.')

    if groq_api:
        st.success('API Key is provided', icon='✅')
    else:
        st.error('API Key not found', icon='🚨')

    st.subheader('Models ֎ and Parameters 🛠️')

    selected_model = st.selectbox('Choose the Model', ["Llama 3 70b", 'Llama 3 8b', 'Llama 3.1 8b instruct'])

    model = None
    if selected_model == 'Llama 3 70b':
        model = 'llama3-70b-8192'
    elif selected_model == 'Llama 3 8b':
        model = 'llama-3.1-8b-instant'
    elif selected_model == 'Llama 3.1 8b instruct':
        model = 'llama-3.1-8b-instant'

    temperature = st.slider('Temperature', min_value=0.01, max_value=1.0, value=0.5, step=0.1)
    p_value = st.slider('P Value', min_value=0.01, max_value=1.0, value=0.9, step=0.1)

# Chat Emojis
bot_emoji = '🤖'
user_emoji = '🙋‍♂️'

# Embeddings function
def load_embeddings():
    persist_directory = 'faissembeddings'

    # Embedding model name
    embeddings_model = 'sentence-transformers/all-mpnet-base-v2'

    # Initialize HuggingFace embeddings
    hf_model = HuggingFaceEmbeddings(model_name=embeddings_model)

    # Load FAISS index (or create if it doesn't exist)

    embeddings_instance = FAISS.load_local(
        folder_path=persist_directory,
        embeddings=hf_model,
        allow_dangerous_deserialization=True  # Enable loading if you trust the source
    )   
    return embeddings_instance

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for embeddings
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = load_embeddings()

# Display chat history
for messages in st.session_state.chat_history:
    with st.chat_message(messages['role']):
        st.markdown(messages['content'])

# Input box for user prompt
user_prompt = st.chat_input('Ask a question about contracts')

# Handle user prompt
if user_prompt:
    st.chat_message('user').markdown(user_prompt)
    st.session_state.chat_history.append({'role': 'user', 'content': user_prompt})

    # Retrieve relevant documents from embeddings
    if st.session_state.embeddings:
        retriever = st.session_state.embeddings.as_retriever()
        retrieved_docs = retriever.get_relevant_documents(user_prompt)

        # Concatenate retrieved documents for context
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Initialize ChatGroq model
        chat_model = ChatGroq(
            api_key=groq_api,
            model=model,
            temperature=temperature,
            top_p=p_value
        )

        # Generate response
        messages = [
            SystemMessage(content="You are a helpful assistant specialized in contracts. If uese asks hi or hello give him generic answer from youself without context  "),
            HumanMessage(content=f"Context: {context}\n\nQuestion: {user_prompt}")
        ]
        response = chat_model(messages)

        # Display response
        st.chat_message('assistant').markdown(response.content)
        st.session_state.chat_history.append({'role': 'assistant', 'content': response.content})
