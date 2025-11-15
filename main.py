from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from qa_system import answer_question  # Import the "brain"

# 1. Initialize your FastAPI app
app = FastAPI(
    title="Aurora AI/ML Take-Home API",
    description="A Q&A system for member messages using RAG.",
    version="1.0.0"
)

# 2. Define the Pydantic models for request (input) and response (output)
class QuestionRequest(BaseModel):
    """The JSON payload for a question."""
    question: str

class AnswerResponse(BaseModel):
    """The JSON response with the answer."""
    answer: str

# 3. Create the /ask API endpoint
@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    """
    Accepts a natural-language question and responds with an answer
    inferred from the member messages.
    """
    if not request.question:
        raise HTTPException(status_code=400, detail="Question field cannot be empty")
    
    print(f"Received question: {request.question}")
    
    # 4. Get the answer from your RAG "brain"
    try:
        answer = answer_question(request.question)
        print(f"Generated answer: {answer}")
        return {"answer": answer}
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# 5. (Optional) A root endpoint to check if the server is running
@app.get("/", include_in_schema=False)
def read_root():
    return {"status": "Aurora QA API is running!", "docs_url": "/docs"}

# 6. This part allows you to run the app with `python main.py`
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)