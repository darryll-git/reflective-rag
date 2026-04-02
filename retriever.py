import os
import time
import argparse
from dotenv import load_dotenv

# Load environment variables from the .env file (specifically GOOGLE_API_KEY)
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# The folder where our lightweight database will be saved
DB_DIR = "faiss_index"

def build_vector_db(pdf_path):
    print(f"Loading document: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    print("Chunking text into readable segments...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    chunks = text_splitter.split_documents(docs)
    
    print(f"Generated {len(chunks)} chunks. Creating embeddings via Google Cloud...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    print("Building and saving FAISS database...")
    vector_db = None
    
    # Trickle-feed batching strategy to avoid burst limits
    batch_size = 30 
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"Embedding batch: Chunks {i+1} to {min(i+batch_size, len(chunks))} of {len(chunks)}...")
        
        # Automatic Retry Logic
        success = False
        retries = 3
        while not success and retries > 0:
            try:
                if vector_db is None:
                    vector_db = FAISS.from_documents(batch, embeddings)
                else:
                    vector_db.add_documents(batch)
                success = True
                
            except Exception as e:
                # If we hit the rate limit, catch the crash, pause, and try again
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"⚠️ Rate limit hit. Backing off for 30 seconds... (Retries left: {retries-1})")
                    time.sleep(30)
                    retries -= 1
                else:
                    raise e # If it is a different error, crash normally
        
        # Short pause between successful batches to keep the API happy
        time.sleep(5)
            
    vector_db.save_local(DB_DIR)
    print("✅ Database built and saved successfully!")
    return vector_db

def get_retriever():
    """Loads the saved database so our agents can search it."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    # allow_dangerous_deserialization is required for local FAISS files in LangChain
    vector_db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    
    # Return a retriever that fetches the top 3 most relevant chunks
    return vector_db.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    # Setup argument parser for terminal execution
    parser = argparse.ArgumentParser(description="Process a PDF and build a FAISS vector database.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file you want to process.")
    parser.add_argument("-q", "--query", type=str, default="What is the main topic of this document?", 
                        help="Optional: A test query to search the database after building it.")
    
    args = parser.parse_args()
    target_pdf = args.pdf_path
    test_query = args.query

    # Check if the file exists
    if os.path.exists(target_pdf):
        print(f"--- Starting Processing for {target_pdf} ---")
        
        # 1. Build the database
        db = build_vector_db(target_pdf)
        
        # 2. Quick Test: Verify the retriever can load and search
        print("\n--- Testing Retriever ---")
        retriever = get_retriever()
        
        results = retriever.invoke(test_query)
        
        print(f"\nQuery: {test_query}")
        if results:
            print(f"Top result found:\n{results[0].page_content[:300]}...")
        else:
            print("No results found.")
            
    else:
        print(f"❌ Error: '{target_pdf}' not found.")
        print("Please ensure the path is correct and try again.")