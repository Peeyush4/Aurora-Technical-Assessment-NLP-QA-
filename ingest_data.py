import json
import os
import chromadb
from chromadb.utils import embedding_functions

# --- Constants ---
# C:\MY FILES\Peeyush-Personal\Coding\Aurora-Technical-Assessment-NLP-QA-\data\
DATA_FILE = "data/response_1762800357568.json"
DB_PATH = "chroma_db"
COLLECTION_NAME = "messages"
EMBED_MODEL = "all-MiniLM-L6-v2"

def main():
    print(f"Loading data from {DATA_FILE}...")
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        items = data.get("items", [])
        if not items:
            print("Error: No 'items' found in the JSON file.")
            return
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {DATA_FILE}")
        return
        
    print(f"Found {len(items)} messages to ingest.")

    # 1. Initialize the embedding model
    print(f"Initializing embedding model: {EMBED_MODEL}...")
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    # 2. Initialize the ChromaDB client (persists to disk)
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
    client = chromadb.PersistentClient(path=DB_PATH)

    # 3. Get or create the collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"} # Use cosine similarity
    )

    # 4. Prepare data for ChromaDB in batches
    batch_size = 100
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        
        documents = []
        metadatas = []
        ids = []

        for item in batch:
            # Skip messages that are empty or just whitespace
            if not item.get("message") or not item["message"].strip():
                continue
                
            message = f"On {item.get('timestamp', 'Unknown date')}, user {item.get('user_name', 'Unknown user')} sent a message: '{item.get('message', '')}'"
            documents.append(message)
            metadatas.append({
                "user_name": item.get("user_name", "Unknown"),
                "user_id": item.get("user_id", "Unknown"),
                "timestamp": item.get("timestamp", "Unknown"),
            })
            ids.append(item["id"])
        
        if not ids:
            continue

        # 5. Add the batch to the collection
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Ingested batch {i//batch_size + 1}/{(len(items)//batch_size) + 1}")
        except Exception as e:
            print(f"Error ingesting batch: {e}")

    print("\n--- Ingestion Complete ---")
    print(f"Total messages in collection: {collection.count()}")

if __name__ == "__main__":
    main()