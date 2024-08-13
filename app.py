# import streamlit as st
# import os
# from dotenv import load_dotenv
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# import re
# import time
# import pdfplumber
# # Load environment variables
# load_dotenv()
# os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
# os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

# # Set up Pinecone
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# persist_directory = 'db/pdf_vector_db'
# sanitized_video_name = re.sub(r'[^a-z0-9\-]', '-', persist_directory.lower())
# persist_directory = sanitized_video_name
# existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

# if persist_directory not in existing_indexes:
#     pc.create_index(
#         name=persist_directory,
#         dimension=768,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )
# while not pc.describe_index(persist_directory).status["ready"]:
#     time.sleep(1)

# # Initialize Pinecone index and embedding model
# index = pc.Index(persist_directory)
# embedding = HuggingFaceEmbeddings()
# pdf_vectordb = PineconeVectorStore(index_name=persist_directory, embedding=embedding)

# # Language models and prompts
# model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
# template = """
# You are a highly intelligent AI assistant. Your primary goal is to provide comprehensive, informative, and accurate responses to user queries, leveraging the provided context. 

# **Guidelines:**

# 1. **Be Clear and Concise:**
#    - Use simple, direct, and precise language.
#     - Structure your responses in a logical and easy-to-follow manner.

#  2. **Avoid Speculation:**
#     - Base your answers strictly on the provided context.
#     - If the context does not provide an answer, politely state that you do not have enough information to respond accurately.

#  3. **Be Helpful:**
#     - If the question is unclear, rephrase it to better understand the user's intent.
#     - Offer additional relevant information or context where appropriate to enhance the user's understanding.

#  4. **Follow ISO Standards:**
#     - Ensure all responses adhere to applicable ISO guidelines for clarity, accuracy, and reliability.
#     - Provide references to ISO standards where applicable to support your responses.

#  5. **Engage Politely:**
#     - only Greet users when they greet you.
#     - Maintain a professional and courteous tone throughout the interaction.

#  *Context:* {context}

#  *Question:* {question}

#  """
# prompt = ChatPromptTemplate.from_template(template)

# def extract_text(file_path):
#     text = ''
#     with pdfplumber.open(file_path) as pdf:
#         for page in pdf.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + '\n'
#     return text

# # Function to embed and store PDF content
# def embed_and_store_pdf(file):
#     global pdf_vectordb
#     document = extract_text(file)
#     docs = Document(page_content=document)
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     text = text_splitter.split_documents([docs])
#     pdf_vectordb = PineconeVectorStore.from_documents(
#         text, embedding, index_name=persist_directory
#     )

# # Function to get answer from chatbot
# def get_answer(question):
#     pdf_vectordb = PineconeVectorStore(index_name=persist_directory,
#                       embedding=embedding)
#     pdf_retriever = pdf_vectordb.as_retriever()
#     print(".................")
#     chain = (
#         {"context": pdf_retriever, "question": RunnablePassthrough()}
#         | prompt
#         | model
#         | StrOutputParser()
#     )
#     answer = chain.invoke(question)
#     print("Res:",answer)
#     return answer

# # Streamlit app
# st.title("Chatbot with PDF Uploads")

# # Sidebar for file uploads
# st.sidebar.header("Upload your PDF")
# uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# # Store chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Handle user input and display response
# if user_input := st.chat_input("What's on your mind?"):
#     st.session_state.messages.append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.markdown(user_input)
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             response = get_answer(user_input)
#             st.markdown(response)
#     st.session_state.messages.append({"role": "assistant", "content": response})





import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re
import time
import pdfplumber

# Load environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

# Set up Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
persist_directory = 'iso-details'
sanitized_video_name = re.sub(r'[^a-z0-9\-]', '-', persist_directory.lower())
persist_directory = sanitized_video_name
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if persist_directory not in existing_indexes:
    pc.create_index(
        name=persist_directory,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
while not pc.describe_index(persist_directory).status["ready"]:
    time.sleep(1)

# Initialize Pinecone index and embedding model
index = pc.Index(persist_directory)
embedding = HuggingFaceEmbeddings()
pdf_vectordb = PineconeVectorStore(index_name=persist_directory, embedding=embedding)

# Language models and prompts
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5)
template = """
You are a highly intelligent AI assistant. Your primary goal is to provide comprehensive, informative, and accurate responses to user queries. 

**Guidelines:**

1. **Be Clear and Concise:**
   - Use simple, direct, and precise language.
    - Structure your responses in a logical and easy-to-follow manner.

 2. **Avoid Speculation:**
    - Base your answers strictly on the provided context.
    - If the context does not provide an answer, politely state that you do not have enough information to respond accurately.

 3. **Be Helpful:**
    - If the question is unclear, rephrase it to better understand the user's intent.
    - Offer additional relevant information or context where appropriate to enhance the user's understanding.

 4. **Follow ISO Standards:**
    - Ensure all responses adhere to applicable ISO guidelines for clarity, accuracy, and reliability.
    - Provide references to ISO standards where applicable to support your responses.

 5. **Engage Politely:**
    - only Greet users when they greet you.
    - Maintain a professional and courteous tone throughout the interaction.

 *Context:* {context}

 *Question:* {question}

 """
prompt = ChatPromptTemplate.from_template(template)

def extract_text(file_path):
    text = ''
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    print("Text Extracted")
    return text

# Function to embed and store PDF content
def embed_and_store_pdf(file):
    global pdf_vectordb
    document = extract_text(file)
    docs = Document(page_content=document)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text = text_splitter.split_documents([docs])
    pdf_vectordb = PineconeVectorStore.from_documents(
        text, embedding, index_name=persist_directory
    )
    print("Vectorized")

# Function to get answer from chatbot
def get_answer(question):
    persist_directory='iso-details'
    sanitized_video_name = re.sub(r'[^a-z0-9\-]', '-', persist_directory.lower())
    persist_directory = sanitized_video_name
    pdf_vectordb = PineconeVectorStore(index_name=persist_directory,
                      embedding=embedding)
    pdf_retriever = pdf_vectordb.as_retriever()
    print(".................")
    chain = (
        {"context": pdf_retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    answer = chain.invoke(question)
    print("Res:",answer)
    return answer

# Streamlit app
st.title("Chatbot with PDF Uploads")

# Sidebar for file uploads
st.sidebar.header("Upload your PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# Process the uploaded file
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Embed and store the PDF content
    embed_and_store_pdf("temp.pdf")
    st.sidebar.success("PDF content has been successfully embedded and stored.")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input and display response
if user_input := st.chat_input("Ask about ISO!"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_answer(user_input)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
