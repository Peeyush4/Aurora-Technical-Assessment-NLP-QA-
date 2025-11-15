import re
import spacy
from fuzzywuzzy import process
# Import the shared database collection
from core.db import collection
# Import the "switched" generator model
from generators import generator

KNOWN_USER_NAMES = [
    'Thiago Monteiro', 'Armand Dupont', "Lily O'Sullivan",
    'Fatima El-Tahir', 'Sophia Al-Farsi', 'Layla Kawaguchi',
    'Amina Van Den Berg', 'Lorenzo Cavalli', 'Vikram Desai',
    'Hans Müller'
]

# --- Helper Function for Name Extraction ---
def extract_user_name(question: str) -> str | None:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(question)
    users_in_question = []
    # Search for named entities first
    for ent in doc.ents:
        if ent.label_ == "PERSON" or ent.label_ == "ORG":
            name = ent.text.strip()
            found_name = process.extractOne(
                name, KNOWN_USER_NAMES, score_cutoff=60
            )
            if found_name:   
                users_in_question.append(found_name[0])
    # Search for proper nouns as well (to catch missed names)
    for ent in doc:
        if ent.pos_ == "PROPN":
            name = ent.text.strip()
            found_name = process.extractOne(
                name, KNOWN_USER_NAMES, score_cutoff=60
            )
            if found_name:
                users_in_question.append(found_name[0])
    return list(set(users_in_question))

# --- User Profile Information ---
def get_user_profiles(user_names: list[str]) -> dict[str, dict]:
    path = "profiles"
    profiles = {}
    for user in user_names:
        # first_name = user.lower().split()[0]
        profiles[user] = open(f"{path}/{user}_mistral_latest.txt").read()
    return profiles

# --- The Core RAG Function ---
def get_rag_information(question: str) -> str:
    if collection is None:
        return "Error: RAG system is not initialized."

    # 1. Retrieve relevant messages
    try:
        results = collection.query(
            query_texts=[question], 
            n_results=30,  # Get 30 relevant messages
            # where=query_filter
        )
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return "Error: Could not retrieve information from the database."

    documents = results.get('documents', [[]])[0]
    if not documents:
        return None
        
    # 2. Build the context string
    context = "Here is the relevant information I found:\n"
    for i, doc in enumerate(documents):
        context += f"- {doc}\n" 
    return context

# --- Main QA Function ---
def answer_question(question: str, using_rag=False) -> str:
    context = ""
    if using_rag:
        context = get_rag_information(question)
        if context is None:
            return "I could not find any relevant information for that query."
    else:
        # 1. Extract user names from the question
        user_names = extract_user_name(question)
        if not user_names:
            return "I do not have that information."

        # 2. Get user profiles
        profiles = get_user_profiles(user_names)

        # 3. Build context from profiles
        context = "Here is the relevant profile information:\n"
        for name, profile in profiles.items():
            context += f"\n--- Profile of {name} ---\n{profile}\n"

    # Build the final prompt
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
    return answer