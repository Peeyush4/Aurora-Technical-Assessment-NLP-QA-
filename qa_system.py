import os
import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions

# --- Load API Key ---
try:
    from dotenv import load_dotenv
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
    genai.configure(api_key=API_KEY)
except ImportError:
    raise ImportError("python-dotenv is not installed. Please run 'pip install python-dotenv'")

# --- Constants ---
DB_PATH = "chroma_db"
COLLECTION_NAME = "messages"
EMBED_MODEL = "all-MiniLM-L6-v2"
MODEL = "gemini-2.5-flash"

# --- Initialize Models and DB on module load ---

# 1. Initialize Gemini Model
print("Initializing Gemini model...")
generation_config = {
  "temperature": 0.2,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 1024,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
gemini_model = genai.GenerativeModel(
    model_name=MODEL,
    generation_config=generation_config,
    safety_settings=safety_settings
)

# 2. Initialize Embedding Function
print(f"Initializing embedding model: {EMBED_MODEL}...")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

# 3. Connect to the existing ChromaDB
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

# --- The Core RAG Function ---

def get_rag_answer(question: str) -> str:
    """
    Retrieves an answer from the RAG system.
    """
    if collection is None:
        return "Error: RAG system is not initialized. Please run ingest_data.py."

    # 1. Retrieve relevant messages from ChromaDB
    # We query for 5 results to get enough context.
    try:
        results = collection.query(
            query_texts=[question],
            n_results=5,
            # (Optional) TODO: Add a `where` filter if you extract a user_name
        )
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return "Error: Could not retrieve information from the database."

    documents = results.get('documents', [[]])[0]
    metadatas = results.get('metadatas', [[]])[0]

    if not documents:
        return "I could not find any relevant information about that."

    # 2. Build the context string
    context = "Here is the information I found:\n"
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        context += f"Message {i+1} (from {meta.get('user_name', 'Unknown')} on {meta.get('timestamp', 'N/A')}):\n"
        context += f'"{doc}"\n\n'

    # 3. Build the prompt for Gemini
    prompt_template = f"""
    You are a professional assistant. Your task is to answer the user's question based *only* on the provided context.
    Do not use any information you were not given. If the answer is not in the context, say "I do not have that information."

    **CONTEXT:**
    {context}

    **QUESTION:**
    {question}

    **ANSWER:**
    """
    
    # 4. Call the Gemini API
    try:
        response = gemini_model.generate_content(prompt_template)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Error: Could not generate an answer."