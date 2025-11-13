import chromadb
from chromadb.utils import embedding_functions

# --- Constants ---
DB_PATH = "chroma_db"
COLLECTION_NAME = "messages"
EMBED_MODEL = "all-MiniLM-L6-v2"

# 1. Initialize Embedding Function
print(f"Initializing embedding model: {EMBED_MODEL}...")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

# 2. Connect to the existing ChromaDB
print(f"Connecting to ChromaDB at {DB_PATH}...")
try:
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )
    print(f"Successfully connected to collection '{COLLECTION_NAME}' with {collection.count()} messages.")
except Exception as e:
    print(f"Error connecting to ChromaDB: {e}")
    print("Please make sure you have run 'python ingest_data.py' first!")
    collection = None