import os
import json
import spacy
from fuzzywuzzy import process, fuzz
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field # Use Pydantic v1 for LangChain tool compatibility
from typing import List

# Import the retriever we built in db.py
from core.db import retriever

# --- Tool 1: The "Smart Name" Finder (spaCy + Fuzz) ---

# --- Load Models at Startup ---
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy NER model loaded successfully.")
except IOError:
    print("FATAL: spaCy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    nlp = None

# This is the "ground truth" list of correct names
KNOWN_USER_NAMES = [
    'Thiago Monteiro', 'Armand Dupont', "Lily O'Sullivan",
    'Fatima El-Tahir', 'Sophia Al-Farsi', 'Layla Kawaguchi',
    'Amina Van Den Berg', 'Lorenzo Cavalli', 'Vikram Desai',
    'Hans Müller'
]

class FindUserNamesInput(BaseModel):
    question: str = Field(description="The user's question mentioning a person")

@tool(args_schema=FindUserNamesInput)
def find_user_names(question: str) -> List[str]:
    """
    Finds the most likely full user name mentioned in a question.
    Use this first to identify *who* the user is asking about.
    Handles typos like 'Amona' or 'Vikrem'.
    """
    if nlp is None:
        return "Error: spaCy NER model not loaded."

    doc = nlp(question)
    
    # Stage 1: Get all potential names (PERSON, ORG, PROPN)
    entity_names = [ent.text for ent in doc.ents if ent.label_ in ("PERSON", "ORG")]
    proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]
    potential_names = []
    if not entity_names and not proper_nouns:
        potential_names = question.lower().replace("?", "").split()
    names_list = set(entity_names + proper_nouns + potential_names)
    
    # Stage 2: Find the best fuzzy match
    users_in_question = []
    for name in names_list:
        possible_names_with_confidence = process.extractBests(
            name,
            KNOWN_USER_NAMES,
            score_cutoff=70,
            limit=5
        )
        users_in_question.extend(
            [name for name, _ in possible_names_with_confidence]
        )
    if users_in_question:
        return list(set(users_in_question))
    return "Error: No user name found in question."


# --- Tool 2: The "RAG" Message Search ---

class RAGSearch(BaseModel):
    user_names: List[str] = Field(description="The full, correct user_name (e.g., 'Vikram Desai').")
    query: str = Field(description="The semantic search query (e.g., 'concert package' or 'seat preference').")

@tool(args_schema=RAGSearch)
def search_messages(user_names: List[str], query: str) -> List[str]:
    """
    Searches the message database for messages from a specific user
    that are semantically related to a query.
    """
    if retriever is None:
        return ["Error: Retriever not initialized."]

    print(f"--- Tool: search_messages(user_names='{user_names}', query='{query}') ---")

    # This is the 10x step: we filter the RAG search by the *user_names*
    # This is a "Metadata Filter"
    rag_result = []
    for user_name in user_names:
        filtered_retriever = retriever.vectorstore.as_retriever(
            search_kwargs={
                "k": 10,
                "filter": {"user_name": user_name}
            }
        )
        # Run the RAG search
        results = filtered_retriever.invoke(query)
        rag_result.extend([doc.page_content for doc in results])

    # Return a clean list of message strings
    return rag_result

class UserRAG(BaseModel):
    question: str = Field(description="The original user question needing message search.")

@tool(args_schema=UserRAG)
def find_user_messages(question: str) -> List[str]:
    """Helper function to call search_messages tool."""
    # `find_user_names` and `search_messages` are tool objects (decorated with @tool)
    # Tool objects are not directly callable like plain functions. Use their
    # `.invoke()` method to execute them programmatically.
    try:
        users_in_question = find_user_names.invoke({"question": question})
    except Exception as e:
        return [f"Error: find_user_names tool invocation failed: {e}"]

    if not isinstance(users_in_question, list):
        return ["Error: Could not find user names."]

    try:
        return search_messages.invoke({"user_names": users_in_question, "query": question})
    except Exception as e:
        return [f"Error: search_messages tool invocation failed: {e}"]

@tool
def get_system_stats() -> dict:
    """
    Returns system statistics like number of users and messages.
    Use this for "meta" questions like "How many users are there?".
    """
    if retriever is None:
        return {"error": "Retriever not initialized."}

    num_users = len(KNOWN_USER_NAMES)
    num_messages = retriever.vectorstore._collection.count()  # Accessing private member for demo purposes

    stats = {
        "number_of_users": num_users,
        "number_of_messages": num_messages,
        "users": KNOWN_USER_NAMES
    }
    return stats

class GetUserRAG(BaseModel):
    question: str = Field(description="The original user question needing message search.")

@tool(args_schema=GetUserRAG)
def get_user_messages(question: str) -> List[str]:
    """
    Search messages related to the `question` for matching user(s) and return
    a list of message strings (RAG results). This tool performs name
    extraction, fuzzy matching against `KNOWN_USER_NAMES`, then runs a
    retriever search filtered by the matched user name(s).

    Returns a list of strings containing message contents. If no user is
    identified, returns an error string.
    """
    if nlp is None:
        return ["Error: spaCy NER model not loaded."]

    doc = nlp(question)
    
    # Stage 1: Get all potential names (PERSON, ORG, PROPN)
    entity_names = [ent.text for ent in doc.ents if ent.label_ in ("PERSON", "ORG")]
    proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]
    potential_names = []
    if not entity_names and not proper_nouns:
        potential_names = question.lower().replace("?", "").split()
    names_list = set(entity_names + proper_nouns + potential_names)
    
    # Stage 2: Find the best fuzzy match
    users_in_question = []
    for name in names_list:
        possible_names_with_confidence = process.extractBests(
            name,
            KNOWN_USER_NAMES,
            score_cutoff=70,
            limit=5
        )
        users_in_question.extend(
            [name for name, _ in possible_names_with_confidence]
        )
    if not users_in_question:
        return "Error: No user name found in question."
    user_names = list(set(users_in_question))

    rag_result = []
    for user_name in user_names:
        filtered_retriever = retriever.vectorstore.as_retriever(
            search_kwargs={
                "k": 10,
                "filter": {"user_name": user_name}
            }
        )
        # Run the RAG search
        results = filtered_retriever.invoke(question)
        rag_result.extend([doc.page_content for doc in results])

    # Return a clean list of message strings
    return rag_result

# --- This list is exported to the agent ---
all_tools = [get_user_messages, get_system_stats]