import streamlit as st
import PyPDF2
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
# Title of the app
st.title("PDF Text Extractor")
load_dotenv()

# Configure Google Generative AI
from google.generativeai import configure
configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Upload PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Check if the user has uploaded a file
if uploaded_file is not None:
    # Read the file into memory
    pdf_file = BytesIO(uploaded_file.read())

    # Create a PDF reader object
    reader = PyPDF2.PdfReader(pdf_file)
    
    # Initialize a variable to store extracted text
    text = ""
    
    # Extract text from each page
    for page in reader.pages:
        text += page.extract_text()
    
    # Display the extracted text
    if text:
         # Split text into documents
        text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=1000,
        chunk_overlap=100
        )
        docs = text_splitter.split_text(text)
        st.subheader("Extracted Text")
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embed=embeddings.embed_query(doc for doc in docs)
        
        st.text_area("Text from PDF", embed, height=500)
    else:
        st.warning("No text found in the PDF.")
