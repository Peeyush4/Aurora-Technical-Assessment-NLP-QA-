import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic.v1 import BaseModel # Use v1 for LangChain compatibility
from langchain_core.messages import HumanMessage
import uuid

# Import our compiled agent
from agent import app 
# Import the spaCy loader
from tools import nlp, KNOWN_USER_ALIASES

# --- API Setup ---
api = FastAPI(
    title="Aurora AI/ML Take-Home API",
    description="A 10x Engineer's Agentic RAG API using LangGraph and Ollama.",
    version="1.0.0"
)

# --- Pydantic Models ---
class QuestionRequest(BaseModel):
    question: str
    # We no longer need thread_id, as our new graph is not conversational
    # thread_id: str = None 

class AnswerResponse(BaseModel):
    answer: str
    # thread_id: str

# --- Startup Event ---
@api.on_event("startup")
def startup_event():
    """
    Runs once when the API starts.
    This just confirms our models are loaded.
    """
    print("--- API Startup ---")
    if nlp is None:
        print("!!! FATAL: spaCy model not loaded. API will fail.")
    else:
        print(f"spaCy NER and {len(KNOWN_USER_ALIASES)} user aliases are loaded.")
    print("--- API Ready ---")


# --- The /ask Endpoint ---
@api.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    """
    Accepts a natural-language question and responds with an answer
    inferred by the LangGraph agent.
    """
    if not request.question:
        raise HTTPException(status_code=400, detail="Question field cannot be empty")
    
    # Our new graph is not conversational, so we don't need a thread_id
    # thread_id = request.thread_id or str(uuid.uuid4())
    # config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n--- New Request ---")
    print(f"Question: {request.question}")

    # 2. This is the 10x step: we run the whole agent
    # This will take 20-30s, as you're fine with.
    try:
        # We pass the question into the 'question' key of our state
        final_state = app.invoke(
            {"question": request.question, "messages": []} 
        )
        
        # The final answer is the last message in the state
        answer = final_state['messages'][-1].content
        
        print(f"Final Answer: {answer}")
        return {"answer": answer} # Removed thread_id
        
    except Exception as e:
        print(f"!!! AGENT FAILED: {e}")
        raise HTTPException(status_code=500, detail=f"Agent Error: {e}")

# --- Root Endpoint ---
@api.get("/", include_in_schema=False)
def read_root():
    return {"status": "Aurora QA API (LangGraph Agent) is running!", "docs_url": "/docs"}

# --- Uvicorn Runner ---
if __name__ == "__main__":
    # Note: This is now 'app.main:api' because it's inside the 'app' folder
    uvicorn.run("main:api", host="0.0.0.0", port=8000, reload=True)