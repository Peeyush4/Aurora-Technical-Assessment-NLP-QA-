import re
import spacy
from fuzzywuzzy import process
# Import the shared database collection
from core.db import retriever
# Import the "switched" generator model
from generators import generator

nlp = spacy.load("en_core_web_sm")
KNOWN_USER_NAMES = [
    'Thiago Monteiro', 'Armand Dupont', "Lily O'Sullivan",
    'Fatima El-Tahir', 'Sophia Al-Farsi', 'Layla Kawaguchi',
    'Amina Van Den Berg', 'Lorenzo Cavalli', 'Vikram Desai',
    'Hans Müller'
]

# --- Helper Function for Name Extraction ---
def extract_user_name(question: str) -> str | None:
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

# --- User Profile Information ---
def get_user_profiles(user_names: list[str]) -> dict[str, dict]:
    path = "profiles"
    profiles = {}
    for user in user_names:
        # first_name = user.lower().split()[0]
        profiles[user] = open(f"{path}/{user}_mistral_latest.txt").read()
    return profiles

# --- The Core RAG Function ---
def get_rag_information(user_names, question: str) -> str:
    """
    Searches the message database for messages from a specific user
    that are semantically related to a query.
    """
    if retriever is None:
        return ["Error: Retriever not initialized."]

    print(f"--- Tool: search_messages(user_names='{user_names}', query='{question}') ---")

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
        results = filtered_retriever.invoke(question)
        rag_result.extend([doc.page_content for doc in results])
        
    # 2. Build the context string
    context = "Here is the relevant information I found:\n"
    for i, doc in enumerate(rag_result):
        context += f"- {doc}\n" 
    return context

# --- Main QA Function ---
def answer_question(question: str, using_rag=True, allow_inference: bool = True) -> str:
    context = ""
    user_names = extract_user_name(question)
    if using_rag:
        context = get_rag_information(user_names, question)
        if context is None:
            return "I could not find any relevant information for that query."
    else:
        # 1. Extract user names from the question
        if not user_names:
            return "I do not have that information."

        # 2. Get user profiles
        profiles = get_user_profiles(user_names)

        # 3. Build context from profiles
        context = "Here is the relevant profile information:\n"
        for name, profile in profiles.items():
            context += f"\n--- Profile of {name} ---\n{profile}\n"

    # Build the final prompt. If `allow_inference` is True, we include extra
    # instructions allowing clearly-labeled speculative inferences.
    base_prompt = (
        """
    You are a professional assistant. Your task is to answer the user's question using ONLY the provided CONTEXT. Treat the CONTEXT as authoritative and complete for the purposes of this answer.

    OUTPUT FORMAT (must follow exactly):
    <Answer text>
    Evidences:
    <evidence line 1>
    <evidence line 2>
    ...

    IMPORTANT RULES (must follow exactly):
    1) Do NOT use or invent any information that is NOT present in the CONTEXT.
    2) The Evidences section must list one or more exact context lines (verbatim) that support the Answer. Each evidence must appear on its own line after the `Evidences:` header.
    3) If the answer is not present in the CONTEXT, respond exactly with the sentence: "I do not have that information." and nothing else.
    4) Be concise. Keep the final answer to 1 sentence plus the Evidences section when applicable.

    """
    )

    inference_instructions = (
        """
    You MAY include an optional `Inferences:` section placed BEFORE `Evidences:`. Each inference line MUST be prefixed with `INFERRED:` to mark it as speculative (for example: `INFERRED: Likely owns a car because they mention taking it to car service.`). Inferences are allowed but must be concise and clearly labeled. Evidence lines are still required and must come after `Evidences:`.

    """
    )

    prompt_body = base_prompt + (inference_instructions if allow_inference else "")

    prompt_template = f"""{prompt_body}\n**CONTEXT:**\n{context}\n\n**QUESTION:**\n{question}\n\n**ANSWER (follow rules above):**\n"""
    
    print("Final Prompt to Generator:\n", prompt_template)
    # 5. Call the generator (This is the pluggable part!)
    response_text = generator.generate(prompt_template)

    # --- Post-check: validate evidence lines exist verbatim in the context ---
    # Accepted formats:
    # 1) <Answer>\nEvidences:\n<e1>\n<e2>...
    # 2) <Answer>\nInferences:\nINFERRED: ...\n...\nEvidences:\n<e1>\n<e2>...
    try:
        import re

        # Find Evidences: header (required)
        evidence_match = re.search(r'(?i)\n\s*evidences?\s*:\s*\n', response_text)
        if not evidence_match:
            return "I do not have that information."

        # Check for optional Inferences: header earlier in the text
        pre_evidence_text = response_text[: evidence_match.start() ]
        inference_match = re.search(r'(?i)\n\s*inferences?\s*:\s*\n', pre_evidence_text)

        if inference_match:
            answer_part = pre_evidence_text[: inference_match.start()].strip()
            inferences_text = pre_evidence_text[ inference_match.end() :].strip()
            inference_lines = [ln.strip() for ln in inferences_text.splitlines() if ln.strip()]
            # Each inference must be labeled with 'INFERRED:'
            for inf in inference_lines:
                if not inf.upper().startswith('INFERRED:'):
                    return "I do not have that information."
        else:
            answer_part = pre_evidence_text.strip()
            inference_lines = []

        # Extract evidences after the Evidences header
        evidence_part = response_text[ evidence_match.end() : ].strip()
        evidence_lines = [line.strip() for line in evidence_part.splitlines() if line.strip()]
        if not evidence_lines:
            return "I do not have that information."

        # Validate each evidence appears verbatim in the retrieved context
        for ev in evidence_lines:
            if ev not in context:
                return "I do not have that information."

        # All checks passed — return the original response trimmed
        return response_text.strip()
    except Exception:
        return "I do not have that information."