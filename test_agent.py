#!/usr/bin/env python
"""Quick script to run the compiled LangGraph agent and print the result.

Usage:
  & "C:/MY FILES/Peeyush-Personal/Coding/.venv/Scripts/Activate.ps1"
  python scripts/test_agent.py "What can you tell me about Thiago Monteiro?"

This script imports the compiled agent 'app' from `app.agent`.
It passes the correct initial state {"messages": [HumanMessage(...)]}
and a config with a thread_id to test the agent's memory.
"""
import sys
from langchain_core.messages import HumanMessage
import uuid

# --- Fix 1: Import from the 'app' directory ---
try:
    # Import the compiled StateGraph 'app' from agent
    from agent import app
except ImportError as e:
    print("Failed to import 'app.agent.app':", e)
    print("Please make sure you are running this from the root 'aurora_qa' directory.")
    raise
except Exception as e:
    print("Failed to import 'app.agent.app':", e)
    raise

def run_question(question: str):
    
    # --- Fix 2: Create the correct state and config ---
    # The new agent's state is a list of messages
    initial_state = {"messages": [HumanMessage(content=question)]}
    
    # The new agent *requires* a thread_id in its config for memory
    # We'll use a unique one for each test run
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        print(f"Invoking agent with question: {question!r}")
        print(f"Using Thread ID: {thread_id}")
        
        # --- Fix 3: Call invoke with both state and config ---
        final_state = app.invoke(initial_state, config=config)

        # This print logic is still correct!
        if isinstance(final_state, dict) and "messages" in final_state:
            last = final_state["messages"][-1]
            content = getattr(last, "content", None)
            if content is not None:
                print("\n--- Final Answer ---\n", content)
            else:
                print("\n--- Final Answer (raw message) ---\n", repr(last))

        # How can I know middle states messages?
        print("\n--- All Messages in Final State ---")
        for i, msg in enumerate(final_state["messages"]):
            print(f"[{i}] {repr(msg)}")

    except Exception as e:
        print("Agent invocation failed:", type(e).__name__, str(e))
        raise


if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        # Let's use your "hard test case" as the default
        # q = "How many users are there in the system?"
        # q = "Get me messages from Thiago Monteiro?"
        # q = "What tools do you have and can you use find_user_messages tool with the question 'Get me the phone number of Thiago Monteiro?' and show me the answer?"
        q = "Get me the phone number of Thiago Monteiro?"
        # q = "What tools do you have?"
        # q = "What does system_messages tool do?"
        # q = "What is Vikram Desai's seat preference for concerts?"

    run_question(q)