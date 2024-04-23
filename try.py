from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama

from PyPDF2 import PdfReader
from dotenv import load_dotenv
import streamlit as st
import os
import time




pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)


folder_path = 'data1'
os.makedirs(folder_path, exist_ok=True) 
for pdf in pdf_docs:
    st.write(pdf)
    file_path =os.path.join(folder_path, pdf.name)
     # Open a file in binary write mode
    with open(file_path, "wb") as f:
        # Write the contents of the uploaded file to the new file.
        f.write(pdf.getbuffer())


   
