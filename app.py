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


    

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')


def load_pdfs(pdf_docs):
    folder_path = 'data'
    os.makedirs(folder_path, exist_ok=True) 
    for pdf in pdf_docs:
        file_path =os.path.join(folder_path, pdf.name)
        # Open a file in binary write mode
        with open(file_path, "wb") as f:
            # Write the contents of the uploaded file to the new file.
            f.write(pdf.getbuffer())
        loader=PyPDFDirectoryLoader("./data")
        documents = loader.load()
    return  documents






def chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_documents(text)
    return chunks



def index(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(text_chunks, embeddings)
    vector.save_local("faiss_index")



def pdf_gpt(human_input):
    llm = ChatOpenAI(model='gpt-4')
    #llm=Ollama(model="llama2") 
    embeddings = OpenAIEmbeddings()
    vector = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    met = vector.similarity_search(human_input, k =4)
    retriever = vector.as_retriever()
    gpt = ConversationalRetrievalChain.from_llm(
        llm, retriever, memory=memory, verbose=True,return_source_documents=True 
    )
    result = gpt.invoke({"question": human_input})
    return result["answer"],result["source_documents"], met



st.title('PDF RAG LLM')
st.header('',divider='rainbow')




with st.sidebar:
    st.title("PDF Loader:")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    if st.button("Submit"):
        with st.spinner("Embedding..."):
            docs = load_pdfs(pdf_docs)
            doc_chunks = chunks(docs)
            index(doc_chunks)
            st.success("Ready to answer queries!")




#make itrator for st.write_stream...For some reasonLLChain are generator not working
def stream_data():
    for word in response.split():
        yield word + " "
        time.sleep(0.02)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"),st.spinner('Please be patient. I am a free open sourced model!'):
        response, rel_doc, mett= pdf_gpt(prompt)
        st.write_stream(stream_data)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    #st.session_state.messages.append({"role": "assistant", "source_documents":rel_doc })




   # With a streamlit expander
    with st.expander("Related Documents"):
        # Find the relevant chunks
        docs =''
        for i, doc in enumerate(rel_doc):
            st.write('**Document**',i+1) #bold lettering
            source = mett[i].metadata['source'][5:] # take out folder name from string
            page = mett[i].metadata['page'] +1 #counts starting at zero
            st.write(f" :book: **page {page} of** ***{source}***") 
            st.write(doc.dict()['page_content'])
            st.write("--------------------------------")
    #st.session_state.messages.append({"role": "assistant", "content": response +'...' '______Source______:' + docs})
