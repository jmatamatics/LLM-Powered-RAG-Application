from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import streamlit as st
import os


    

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

def pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks



def index(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_texts(text_chunks, embeddings)
    vector.save_local("faiss_index")



def pdf_gpt(human_input):
    llm = ChatOpenAI(model='gpt-4') 
    embeddings = OpenAIEmbeddings()
    vector = FAISS.load_local("faiss_index", embeddings)
    retriever = vector.as_retriever()
    gpt = ConversationalRetrievalChain.from_llm(
        llm, retriever, memory=memory, verbose=False,return_source_documents=True 
    )
    result = gpt.invoke({"question": human_input})
    return result["answer"],result["source_documents"][0].dict()['page_content']



st.title('PDF RAG LLM')
input_text= st.text_input("What would you like to know?")


if input_text:
    answer, rel_doc= pdf_gpt(input_text)
    st.session_state.clear()
    st.write(answer) 
    st.header('Source', divider='rainbow')
    st.write(rel_doc) 



with st.sidebar:
    st.title("PDF Loader:")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    if st.button("Submit"):
        with st.spinner("Embedding..."):
            text = pdf_text(pdf_docs)
            text_chunks = chunks(text)
            index(text_chunks)
            st.success("Ready to answer queries!")
