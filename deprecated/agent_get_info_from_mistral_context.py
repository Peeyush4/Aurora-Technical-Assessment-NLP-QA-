import os
import re
import json
import uuid
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# We import all the tools for the "brain" to use
from tools import all_tools

# --- 1. Define the Agent's "Memory" (State) ---
# We go back to the simple, standard, powerful state
class AgentState(TypedDict):
    # This holds the full chat history
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# --- 2. Define the Agent's "Brain" (The LLM) ---

# This is the NEW, 10x-ENGINEER "ReAct" (Reason+Act) PROMPT.
# This prompt is *much* more direct and will fix the "thinking out loud" bug.
AGENT_SYSTEM_PROMPT = """
You are an expert assistant. You have a "toolbox" to answer questions. 
You must answer the user's question by *thinking step-by-step* and *using the tools*.
You must follow a strict "Reason-Act" loop.
You MUST use the tools when needed and are stuck. 
You MUST NOT answer questions without using the tools if the question requires specific data.
You can use tools multiple times if needed.

**TOOLS:**
You have access to the following tools. You must read their descriptions to form your plan.

1.  `get_system_stats()`:
    - **Description:** Use this for "meta" questions like "How many users are there?" or "What are the names of all users?".
    - **Arguments:** None.

2.  `find_user_messages(question: str)`:
    - **Description:** This tool searches the database for messages from a user extracted from a about a *specific query*.
    - **Arguments:** 
        - **question (str):** The user's original question (e.g., "What about Vikrem and Amona?").
    - **Returns:** A list of message strings related to that user and query.

**YOUR TASK & PLANNING HINTS:**
You must *decide* which tool to call based on the user's *latest* message.
-   **If the question is "meta"** (like "How many users?"), you MUST call `get_system_stats`.
-   **If the question is about a specific person** (like "What is Thiago's phone number?"):
    1.  You **FIRST** step is to call `find_user_messages` using that question (e.g., "What is Thiago's phone number?").
    2.  Once you have the messages, you can form the final answer.
-   **If you have all the information** from previous tool calls, you must formulate the final answer.

**CRITICAL RULES:**
- **DEFAULT BEHAVIOR:** Your default behavior is to be concise. **DO NOT** "think out loud" or talk about your plan. Just call the tool or give the final answer.
- **DEBUG MODE:** If the user *explicitly asks* for "reasoning", "your plan", or "how you work", you *should* explain your step-by-step plan *as* your final answer.
- **DO NOT** respond in prose *and* call a tool. If you call a tool, your *entire* response must be the tool call.

**REASONING RULES (for when you formulate a final answer):**
(These are the same as before and are critical)
- **Contradiction Rule:** If tool outputs are contradictory (e.g., "window seat" vs. "aisle seat"), the message with the **latest timestamp** is the correct one.
- **"No Car" Rule:** If you search for "owns a car" and only find "car service," the answer is "0 cars owned."
- **Alias Rule:** You must connect feedback to events (e.g., "concert package" *is* feedback for the "son's graduation").
- **Unstated Info Rule:** If the user asks for info not present in any message (e.g., "phone number"), respond with "Information not found."
"""

# The LLM "Brain"
llm = ChatOllama(model="mistral", temperature=0)
llm_with_tools = llm.bind_tools(all_tools)

# --- 3. Define the "Nodes" (The Agent's Actions) ---

# This node is the "brain"
def call_model_node(state: AgentState):
    """The primary node that calls the LLM (Mistral) to decide what to do."""
    print("--- Node: call_model (Agent Brain) ---")
    
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + state['messages'] 
        
    # This invoke will EITHER return a final answer OR a tool call
    response = llm_with_tools.invoke(messages)
    print(f"  -> LLM Response: {repr(response)}")
    print("tool_calls attr:", getattr(response, "tool_calls", None))
    # Print any invalid tool-calls parsed by the wrapper for debugging
    if getattr(response, "invalid_tool_calls", None):
        print("invalid_tool_calls:", getattr(response, "invalid_tool_calls"))
    print("content/text:", getattr(response, "content", getattr(response, "text", None)))
    print("dir(response):", [n for n in dir(response) if not n.startswith('_')])
    return {"messages": [response]}

# This node runs the tools
def call_tool_node(state: AgentState):
    """This node checks for tool calls and executes them."""
    print("--- Node: call_tool ---")
    
    last_message = state['messages'][-1]
    
    # If the last message contains structured tool_calls, use them.
    tool_calls = getattr(last_message, 'tool_calls', None)

    # Fallback: some LLMs return textual JSON describing a tool call.
    # Try to parse JSON from the message content and normalize it into
    # the expected `tool_calls` shape: list of dicts with keys 'name','args','id'.
    if not tool_calls:
        text = getattr(last_message, 'content', '') or getattr(last_message, 'text', '')
        if text:
            m = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    if isinstance(parsed, dict):
                        parsed = [parsed]
                    normalized = []
                    for item in parsed:
                        name = item.get('name') or item.get('tool')
                        args = item.get('args') or item.get('arguments') or {}
                        normalized.append({'name': name, 'args': args, 'id': item.get('id') or str(uuid.uuid4())})
                    tool_calls = normalized
                except Exception:
                    tool_calls = None

    # If we have tool calls now, execute them
    if tool_calls:
        tool_outputs = []
        tool_map = {t.name: t for t in all_tools}
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            if tool_name not in tool_map:
                raise ValueError(f"Requested tool '{tool_name}' not available in all_tools")
            tool_to_call = tool_map[tool_name]

            print(f"  -> Calling Tool: {tool_name}({tool_call['args']})")
            output = tool_to_call.invoke(tool_call['args'])
            
            tool_outputs.append(
                ToolMessage(content=str(output), tool_call_id=tool_call['id'])
            )
        
        # Return the tool outputs
        return {"messages": tool_outputs}
    else:
        # No tool calls found; nothing to do
        return {}

# --- 4. Define the "Edges" (The Agent's Flowchart) ---
def should_continue(state: AgentState):
    """This edge decides if the agent is done or needs to loop (to think more)."""
    last_message = state['messages'][-1]
    # If structured tool_calls exist, loop to run them
    if getattr(last_message, 'tool_calls', None):
        return "call_tools"

    # Fallback: if the model returned textual JSON describing a tool call
    # (e.g., '[{"name":"find_user_messages","arguments":{...}}]'), detect it
    # and treat it as a tool call so the run_tools node executes.
    text = getattr(last_message, 'content', '') or getattr(last_message, 'text', '')
    if text:
        m = re.search(r"(\[\s*\{\s*\"name\"\s*:\s*\"[A-Za-z0-9_]+\"[\s\S]*\}|\{\s*\"name\"\s*:\s*\"[A-Za-z0-9_]+\"[\s\S]*\})", text, re.DOTALL)
        if m:
            return "call_tools"

    # Otherwise the brain gave a final answer
    return "end"

# --- 5. Build the Graph ---
print("Building 10x Agentic RAG (Cyclical)...")
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("agent_brain", call_model_node)
workflow.add_node("run_tools", call_tool_node)

# Define the flowchart
workflow.set_entry_point("agent_brain")

workflow.add_conditional_edges(
    "agent_brain",
    should_continue,
    {
        "call_tools": "run_tools", # If brain calls tool, run tool
        "end": END                 # If brain gives answer, end
    }
)
# This is the "loop" you wanted
workflow.add_edge("run_tools", "agent_brain") # After running tool, go back to brain to "think"

# Add conversational memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
print("LangGraph Agent compiled successfully.")