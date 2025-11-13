import re
# Import the shared database collection
from core.db import collection
# Import the "switched" generator model
from generators import generator

# --- Helper Function for Name Extraction ---
def extract_user_name(question: str) -> str | None:
    question_lower = question.lower()
    if "amira" in question_lower:
        print("Detected 'Amira', correcting to 'Amina Van Den Berg'")
        return "Amina Van Den Berg"
    # This could be expanded
    return None

# --- The Core RAG Function ---
def get_rag_answer(question: str) -> str:
    if collection is None:
        return "Error: RAG system is not initialized."

    # 1. Try to extract a user name to filter by
    # user_name = extract_user_name(question)
    # query_filter = {}
    # if user_name:
    #     query_filter = {"user_name": user_name}

    # 2. Retrieve relevant messages
    try:
        results = collection.query(
            query_texts=[question],
            n_results=5,  # Get 5 relevant messages
            # where=query_filter
        )
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return "Error: Could not retrieve information from the database."

    documents = results.get('documents', [[]])[0]
    if not documents:
        return "I could not find any relevant information for that query."

    # 3. Build the context string
    context = "Here is the relevant information I found:\n"
    for i, doc in enumerate(documents):
        context += f"- {doc}\n" 

    # 4. Build the final prompt
    prompt_template = f"""
    You are a professional assistant. Your task is to answer the user's question based *only* on the provided context.
    Do not use any information you were not given. If the answer is not in the context, say "I do not have that information."
    Be concise.

    **CONTEXT:**
    {context}

    **QUESTION:**
    {question}

    **ANSWER:**
    """
    
    print("Final Prompt to Generator:\n", prompt_template)
    # 5. Call the generator (This is the pluggable part!)
    answer = generator.generate(prompt_template)
    
    # # Handle "Amira" typo in the final output
    # if "Amina" in answer and "amira" in question.lower():
    #     answer = answer.replace("Amina", "Amira")
            
    return answer