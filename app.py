
from dotenv import load_dotenv
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import os
import time
import shutil
from agents import *

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")




st.title('RAG LLM App using LangChain')
st.subheader('by Joseph Mata')
st.header('',divider='rainbow')

with st.sidebar:
    st.title("PDF Loader")
    pdf_docs = st.file_uploader("", accept_multiple_files=True)
    if pdf_docs is not None:
        for i, pdf in enumerate(pdf_docs):
            # turn into bytes
            bytes_data = pdf.read()
            # Pass a unique key for each widget
            pdf_viewer(bytes_data,width = 100, height =122, pages_to_render =[1], key=f"pdf_viewer_{i}")
#pages_to_render =[1]width = 300, height =342,
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
        if "i'm sorry" or "assist" or "please" not in response.lower():
        # Find the relevant chunks
            for i, doc in enumerate(rel_docs):
                st.write('**Document**',i+1) #bold lettering
                source = rel_docs[i].metadata['source'][5:] # take out folder name from string
                page = rel_docs[i].metadata['page'] +1 #counts starting at zero
                st.write(f" :book: **page {page} of** ***{source}***") 
                st.write(rel_docs[i].page_content)
                pdf_viewer(rel_docs[i].metadata['source'],width = 670, height =790, pages_to_render =[page], key=f"pdf_viewer_{i+100}")

                st.write("--------------------------------")
