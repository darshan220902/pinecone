from dotenv import load_dotenv
import os
import pickle
import time
import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configure Google Generative AI
from google.generativeai import configure
configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Streamlit app
st.title("Hi, I'm Jerry, your Research Bot! 🤖")
st.subheader("Type your query and let's uncover some answers! 🕵️‍♂️")

# Sidebar to input URLs
st.sidebar.title("Add URLs on topics you'd like to explore and see the magic unfold! ✨")
urls = [st.sidebar.text_input(f"URL{i+1}") for i in range(4)]
process = st.sidebar.button("Run 🚀")

file_path = "faiss_store_gemini.pkl"

if process:
    # Load data from URLs
    loader = UnstructuredURLLoader(urls=urls)
    L1=st.text("Data Loading...Started...⌛⌛")
    data = loader.load()
    L1.text("Data Loaded...☑️")

    # Split text into documents
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=1000,
        chunk_overlap=100
    )
    L2=st.text("Text Splitter...Started...⌛⌛")
    docs = text_splitter.split_documents(data)
    L2.text("Text Splitted...☑️")

    # Generate embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    L3=st.text("Embedding Vector Started Building...⌛⌛")
    time.sleep(2)
    L3.text("Embedding Vector Done...☑️")

    # Save vector store
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

# Query input
query = st.text_input(" ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorIndex = pickle.load(f)
            llm_prompt_template = """You are an assistant for question-answering tasks.
            Use the following context to answer the question.
            If you don't know the answer, just say that you don't know.
            Use five sentences maximum and keep the answer concise.\n
            Question: {question} \nContext: {context} \nAnswer:"""
            retriever = vectorIndex.as_retriever(search_kwargs={"k": 6})

            llm_prompt = PromptTemplate.from_template(llm_prompt_template)

            # Chain for processing query
            rag_chain = (
                {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
                 "question": RunnablePassthrough()}
                | llm_prompt
                | ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, top_p=0.85)
                | StrOutputParser()
            )
            result = rag_chain.invoke(query)

            # Display result
            st.subheader("Answer 💡")
            st.write(result)