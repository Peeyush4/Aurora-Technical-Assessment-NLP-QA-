#!/usr/bin/env python
"""Quick script to run the compiled LangGraph agent and print the result.

Usage:
  & "C:/MY FILES/Peeyush-Personal/Coding/.venv/Scripts/Activate.ps1"
  python scripts/test_agent.py "What can you tell me about Thiago Monteiro?"

This script imports the compiled agent object from `agent.py` (renamed to
`agent_app` in `main.py` but here we import directly from `agent`). It passes
a minimal initial state that contains a `question` key and prints the final
state returned by `app.invoke()`.
"""
import sys
import json
from langchain_core.messages import HumanMessage

try:
    # Import the compiled StateGraph 'app' from agent
    from agent import app
except Exception as e:
    print("Failed to import 'agent.app':", e)
    raise

def run_question(question: str):
    # Build the minimal initial state expected by the graph
    initial_state = {"question": question}

    try:
        print(f"Invoking agent with question: {question!r}")
        final_state = app.invoke(initial_state)
        print("\n--- Final State (raw) ---")
        try:
            # Pretty-print any nested structures
            print(json.dumps(final_state, default=str, indent=2))
        except Exception:
            print(repr(final_state))

        # If the agent returned a 'messages' list, try to show the last reply
        if isinstance(final_state, dict) and "messages" in final_state:
            last = final_state["messages"][-1]
            # Many LangGraph/LLM messages expose `.content` on the message object
            content = getattr(last, "content", None)
            if content is not None:
                print("\n--- Final Answer ---\n", content)
            else:
                print("\n--- Final Answer (raw message) ---\n", repr(last))

    except Exception as e:
        print("Agent invocation failed:", type(e).__name__, str(e))
        raise


if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = "What is the phone number for Thiago Monteiro?"

    run_question(q)
