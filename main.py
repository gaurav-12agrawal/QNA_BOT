import os
import pickle
import time
import streamlit as st
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up the title and sidebar UI elements
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Initialize list to store valid URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}").strip()
    if url:
        urls.append(url)

# Button to process the URLs
process_url_clicked = st.sidebar.button("Process URLs")

# Placeholder for main UI feedback
main_placeholder = st.empty()

# Define file path for the FAISS index
file_path = "faiss_store_openai.pkl"

# Set up the Language Model with specific parameters
llm = OpenAI(temperature=0.9, max_tokens=500)

# Process URLs if button is clicked
if process_url_clicked and urls:
    main_placeholder.text("Data Loading...Started...✅✅✅")
    
    # Load data from URLs
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    
    # Split data using specified separators
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitting...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    
    # Create embeddings and build FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding and Index Building...Started...✅✅✅")
    time.sleep(2)  # Artificial delay for user feedback
    
    # Save the FAISS index to a file using pickle
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)
    main_placeholder.text("Data processing complete. Ready to answer questions!")

# Input field for the user to ask a question
query = st.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        # Load the FAISS index from the file
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
        
        # Set up the retrieval and answer chain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        
        # Display the answer and sources
        st.header("Answer")
        st.write(result.get("answer", "No answer available."))
        
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
    else:
        st.error("No indexed data available. Please process some URLs first.")
