from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import os
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')




def load_pdfs(pdf_docs):
    folder_path = 'data'
    os.makedirs(folder_path, exist_ok=True) 
    for pdf in pdf_docs:
        file_path =os.path.join(folder_path, pdf.name)
        with open(file_path, "wb") as f:
            f.write(pdf.getbuffer())
        loader=PyPDFDirectoryLoader("./data")
        documents = loader.load()
    return  documents


def chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(text)
    return chunks



def index(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(text_chunks, embeddings)
    vector.save_local("faiss_index")



def pdf_gpt(human_input):
    llm = ChatOpenAI(model='gpt-4')
    embeddings = OpenAIEmbeddings()
    try:
        vector = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as error:
        vector = FAISS.load_local("Ghuru_index", embeddings, allow_dangerous_deserialization=True)
    rel_docs= vector.similarity_search(human_input, k=4)
    retriever = vector.as_retriever()
    gpt = ConversationalRetrievalChain.from_llm(
        llm, retriever, memory=memory, verbose=True,return_source_documents=False
    )
    result = gpt.invoke({"question": human_input})
    return result["answer"], rel_docs   
