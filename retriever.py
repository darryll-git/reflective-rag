import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# The folder where our lightweight database will be saved
DB_DIR = "faiss_index"

def build_vector_db(pdf_path):
    print(f"Loading document: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    print("Chunking text into readable segments...")
    # We use overlapping chunks to ensure context isn't lost between paragraphs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    print(f"Generated {len(chunks)} chunks. Creating embeddings via Ollama...")
    # This runs locally using the model you just pulled
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    print("Building and saving FAISS database...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(DB_DIR)
    print("✅ Database built and saved successfully!")
    return vector_db

def get_retriever():
    """Loads the saved database so our agents can search it."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # allow_dangerous_deserialization is required for local FAISS files in LangChain
    vector_db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    
    # Return a retriever that fetches the top 3 most relevant chunks
    return vector_db.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    # --- TEST BLOCK ---
    # 1. Place a PDF in your project folder and put its name here:
    test_pdf = "sample.pdf" 
    
    # 2. Uncomment the line below to build the database for the first time:
    build_vector_db(test_pdf)
    
    # 3. Uncomment the lines below to test searching the database:
    # retriever = get_retriever()
    # results = retriever.invoke("What is the main topic of this document?")
    # print(f"\nFound {len(results)} relevant chunks!")
    # for i, res in enumerate(results):
    #     print(f"\n--- Chunk {i+1} ---")
    #     print(res.page_content[:200] + "...")