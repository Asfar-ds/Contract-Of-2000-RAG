from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()
groq_api = os.getenv('GROQ_API_KEY')
os.environ['GROQ_API_KEY'] = groq_api  # Set the environment variable

# Streamlit app configuration
st.set_page_config(page_title='Llama Models for Contracts', page_icon=':robot_face:', layout='wide')

# Sidebar configuration
st.sidebar.title('🏛️💬 AI Legal Assistant')
st.sidebar.write('''The model is trained on 512 contract files, including Real Property Leases, Intellectual Property Assignments, Indebtedness Agreements, Subcontracts, and Transition Services Agreements. Chat with the AI Assistant to explore these contracts.
''')


uploaded_file = st.sidebar.file_uploader('You can also upload your own data', type=['txt', 'pdf'])




def load_and_split_file(file):
    if file is None:
        return None
    else:
        # Save the uploaded file temporarily
        temp_file_path = os.path.join("temp", file.name)
        os.makedirs("temp", exist_ok=True)  # Ensure temp directory exists
        
        with open(temp_file_path, "wb") as f:
            f.write(file.getbuffer())
        
        # Determine the loader based on file type
        if file.type == 'text/plain':
            loader = TextLoader(temp_file_path)
        elif file.type == 'application/pdf':
            loader = PyPDFLoader(temp_file_path)
        else:
            st.error("Unsupported file type.")
            return None

        # Load and split the document
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(docs)

persist_directory = 'faissembeddings'
model_docs = 'sentence-transformers/all-mpnet-base-v2'








if groq_api:
    st.sidebar.success('API Key is provided', icon='✅')
else:
    st.sidebar.error('API Key not found', icon='🚨')

st.sidebar.subheader('Models ֎ and Parameters 🛠️')
selected_model = st.sidebar.selectbox('Choose the Model', ["Llama 3 70b", 'Llama 3 8b', 'Llama 3.1 8b instruct'])

model = None
if selected_model == 'Llama 3 70b':
    model = 'llama3-70b-8192'
elif selected_model == 'Llama 3 8b':
    model = 'llama-3.1-8b-instant'
elif selected_model == 'Llama 3.1 8b instruct':
    model = 'llama-3.1-8b-instant'

temperature = st.sidebar.slider('Temperature', min_value=0.01, max_value=1.0, value=0.5, step=0.1)
p_value = st.sidebar.slider('P Value', min_value=0.01, max_value=1.0, value=0.9, step=0.1)

# Chat Emojis
bot_emoji = '🤖'
user_emoji = '🙋‍♂️'



# Embeddings function
def load_embeddings(modelHF):
    persist_directory = 'faissembeddings'

    # Embedding model name
    embeddings_model = modelHF

    # Initialize Hugging Face embeddings
    hf_model = HuggingFaceEmbeddings(model_name=embeddings_model)


    # Load FAISS index (or create if it doesn't exist)
    embeddings_instance = FAISS.load_local(
        folder_path=persist_directory,
        embeddings=hf_model,
        allow_dangerous_deserialization=True  # Enable loading if you trust the source
    )
    return embeddings_instance

# Initialize session state for embeddings and chat history
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = load_embeddings(model_docs)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []





def add_doc_to_embeddings(file):
    if file is not None:
        documents = load_and_split_file(file)
        
        st.session_state.embeddings.add_documents(documents)
        st.session_state.embeddings.save_local(folder_path=persist_directory)

    # Button to add document
if uploaded_file:
    if st.sidebar.button("Add Document to Embeddings"):
        add_doc_to_embeddings(uploaded_file)




# Display chat history

for messages in st.session_state.get("chat_history", []):
    with st.chat_message(messages['role']):
        st.markdown(messages['content'])

def add_sample_questions():
    st.subheader('Sample Questions:')
    st.write('👉 How do the "364-Day" and "Five-Year" Credit Agreements differ in terms of interest rates, payment, and defaults?')
    st.write("👉 What is the effect of the covenants in these credit agreements on the borrower's operations and finances?")
    st.write('👉 What provisions are made for managing confidential information in these agreements, and what are their potential impacts?')

# Input box for user prompt
user_prompt = st.chat_input('Ask a question about contracts')

if not user_prompt:
    add_sample_questions()

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
            SystemMessage(content="You are a helpful assistant specialized in contracts. If user asks you to answer in another language than only answer in that language. "),
            HumanMessage(content=f"Context: {context}\n\nQuestion: {user_prompt}")
        ]
        response = chat_model(messages)

        # Display response
        st.chat_message('assistant').markdown(response.content)
        st.session_state.chat_history.append({'role': 'assistant', 'content': response.content})


