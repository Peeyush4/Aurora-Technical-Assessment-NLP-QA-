from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Constants ---
DB_PATH = "chroma_db"
COLLECTION_NAME = "messages"
EMBED_MODEL = "all-MiniLM-L6-v2"

# --- Load Models and DB at Startup ---
try:
    print("Loading Embedding Model (for RAG)...")
    embedding_func = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
    )
    
    print(f"Connecting to Vector DB at {DB_PATH}...")
    # Connect to the database we already built with ingest_data.py
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_func,
        collection_name=COLLECTION_NAME
    )
    
    # Create a retriever that will be used by our tools
    # We set k=10 to give the agent *plenty* of clues
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    print("ChromaDB Retriever is ready.")

except Exception as e:
    print(f"!!! FATAL ERROR connecting to ChromaDB: {e}")
    print("!!! --- Have you run 'python ingest_data.py' first? --- !!!")
    retriever = None