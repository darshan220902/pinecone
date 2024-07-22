import streamlit as st
import PyPDF2
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import numpy as np
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
  # Use the correct import statement
from pinecone import ServerlessSpec

# Title of the app
st.title("PDF Text Extractor and Embedding")

# Load environment variables
load_dotenv()

# Configure Google Generative AI
from google.generativeai import configure
configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Pinecone
pinecone_api_key = os.getenv("api_key")


# Initialize Pinecone Client
pc = Pinecone(api_key=pinecone_api_key)

# Define the index name and dimension
index_name = "pdf1-text-index"
dimension = 768  # Update the dimension according to your embedding model's output size

# Check if the index exists, if not create it
if index_name not in pc.list_indexes():
    pc.create_index(name=index_name, dimension=dimension, metric='cosine', spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    ))
index = pc.Index(index_name)

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
        st.subheader("Extracted Text")
        st.text_area("Text from PDF", text, height=300)
        
        # Split text into documents
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " "],
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = text_splitter.split_text(text)

        # Generate embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embeddings_vectors = []
        for doc in docs:
            try:
                vector = embeddings.embed_query(doc)
                embeddings_vectors.append(vector)
            except Exception as e:
                st.error(f"An error occurred while generating embeddings for a document: {str(e)}")
                st.stop()

        # Upsert embeddings to Pinecone
        ids = [str(i) for i in range(len(docs))]
        try:
            index.upsert(vectors=zip(ids, embeddings_vectors))  # Add embeddings to Pinecone index
            st.success("Embeddings have been successfully added to Pinecone!")
        except Exception as e:
            st.error(f"An error occurred while upserting vectors: {str(e)}")
            st.stop()

        # Query Pinecone for similar text
        query_text = st.text_input("Enter text to query Pinecone")
        if query_text:
            try:
                query_vector = embeddings.embed_query(query_text)
                results = index.query(queries=[query_vector], top_k=5)  # Retrieve the top 5 closest matches
                st.subheader("Query Results")
                for match in results['matches']:
                    st.write(f"Document ID: {match['id']}, Score: {match['score']}")
            except Exception as e:
                st.error(f"An error occurred while querying Pinecone: {str(e)}")
                st.stop()
    else:
        st.warning("No text found in the PDF.")
else:
    st.info("Please upload a PDF file to get started.")
