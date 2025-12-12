import streamlit as st
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough 
import os

st.title("NVIDIA NIM RAG DEMO")

user_api_key=st.text_input("Enter your NVIDIA API Key:",type="password")

if not user_api_key:
    st.warning("Please enter your NVIDIA API Key to begin.")
    st.stop()

llm=ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    api_key=user_api_key,
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024
)

uploaded_files = st.file_uploader(
    "Upload one or multiple PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

def create_vector_store(files):
    docs=[]
    for file in files:
        temp_path=f"./temp_{file.name}"
        with open(temp_path,"wb") as f:
            f.write(file.read())

        loader=PyPDFLoader(temp_path)
        docs.extend(loader.load())

        os.remove(temp_path)
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    embeddings = NVIDIAEmbeddings(api_key=user_api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store

if st.button("Create Embeddings"):
    if uploaded_files:
        st.session_state.vector_store = create_vector_store(uploaded_files)
        st.success("FAISS Vector DB created using uploaded PDFs!")
    else:
        st.warning("Please upload at least one PDF file.")

query = st.text_input("Ask a question based on your documents:")

if query:
    if "vector_store" not in st.session_state:
        st.warning("Please create embeddings first.")
    else:
        retriever = st.session_state.vector_store.as_retriever()

        prompt = ChatPromptTemplate.from_template("""
        Answer the questions *only* using the context below.
        <context>
        {context}
        </context>
        Question: {question}
        """)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = rag_chain.invoke(query)

        st.subheader("Answer:")
        st.write(response)