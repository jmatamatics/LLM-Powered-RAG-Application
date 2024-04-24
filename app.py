from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import streamlit as st
import os
import time
import shutil


load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
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




st.title('RAG LLM App using LangChain')
st.subheader('by Joseph Mata')
st.header('',divider='rainbow')


with st.sidebar:
    st.title("PDF Loader")
    pdf_docs = st.file_uploader("", accept_multiple_files=True)
    if st.button("Submit"):
        with st.spinner("Embedding..."):
            docs = load_pdfs(pdf_docs)
            doc_chunks = chunks(docs)
            index(doc_chunks)
            st.success("Ready to answer queries!")
    st.header('',divider='rainbow')

    if st.button("Delete Uploaded Files"):
        # Remove the directory and all its contents
        try:
            shutil.rmtree('./data')
            print('')
            shutil.rmtree('./faiss_index')
            print('')
            shutil.rmtree('./data')
            st.success("Files deleted!")
        except Exception as error:
            st.success("Files deleted!")


## code adapted from streamlit.io ##
#make itrator for st.write_stream.
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
if prompt := st.chat_input("What's your question?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"),st.spinner("A man who is a master of patience is master of everything else. ~ George Savile"):
        response, rel_docs= pdf_gpt(prompt)
        st.write_stream(stream_data)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})



   # With a streamlit expander
    with st.expander("Related Documents"):
        # Find the relevant chunks
        for i, doc in enumerate(rel_docs):
            st.write('**Document**',i+1) #bold lettering
            source = rel_docs[i].metadata['source'][5:] # take out folder name from string
            page = rel_docs[i].metadata['page'] +1 #counts starting at zero
            st.write(f" :book: **page {page} of** ***{source}***") 
            st.write(rel_docs[i].page_content)
            st.write("--------------------------------")
