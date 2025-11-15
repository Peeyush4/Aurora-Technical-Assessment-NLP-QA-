import os
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
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
Your goal is to get a final answer for the user.

**TOOLS:**
You have three tools:
1.  `get_system_stats()`: Use for "meta" questions like "How many users are there?"
2.  `find_user_names(question)`: Use *first* for any question about a specific person (e.g., "Vikram", "Amona").
3.  `search_messages(user_name, query)`: Use *after* you have a user's name to get context about them.

**YOUR REASONING PROCESS (A Loop):**
1.  **Analyze the user's *latest* message.**
2.  **Plan:** Decide *one* of two actions:
    a) **Call a tool:** If you need information (e.g., find a user, search messages, get stats).
    b) **Respond to user:** If you have *all* the information needed from previous tool calls.
3.  **Act:** If you decide to call a tool, call it. If you decide to respond, formulate the answer.

**CRITICAL RULES FOR FORMULATING ANSWERS:**
- **NEVER** answer a question about a user (e.g., "Vikram's cars") unless you have *already* called `search_messages` and have the context.
- **Contradiction Rule:** If tool outputs are contradictory (e.g., "window seat" vs. "aisle seat"), the message with the **latest timestamp** is the correct one.
- **Reasoning Rule:** You must connect feedback to events (e.g., "concert package" *is* feedback for the "son's graduation").
- **"No Car" Rule:** If you search for "owns a car" and only find "car service," the answer is "0 cars owned."
- **"Thinking Out Loud":** Do NOT "think out loud" or talk about your plan. Just call the tool or give the final answer.
"""

# The LLM "Brain"
llm = ChatOllama(model="mistral", temperature=0)
llm_with_tools = llm.bind_tools(all_tools)

# --- 3. Define the "Nodes" (The Agent's Actions) ---

# This node is the "brain"
def call_model_node(state: AgentState):
    """The primary node that calls the LLM (Mistral) to decide what to do."""
    print("--- Node: call_model (Agent Brain) ---")
    
    messages = state['messages']
    
    # Add the system prompt *only* if it's the first message
    if len(messages) == 1:
        messages = [HumanMessage(content=AGENT_SYSTEM_PROMPT)] + messages
        
    # This invoke will EITHER return a final answer OR a tool call
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}

# This node runs the tools
def call_tool_node(state: AgentState):
    """This node checks for tool calls and executes them."""
    print("--- Node: call_tool ---")
    
    last_message = state['messages'][-1]
    
    # If the last message is a tool call, run it
    if last_message.tool_calls:
        tool_outputs = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            tool_to_call = {t.name: t for t in all_tools}[tool_name]
            
            print(f"  -> Calling Tool: {tool_name}({tool_call['args']})")
            output = tool_to_call.invoke(tool_call['args'])
            
            tool_outputs.append(
                ToolMessage(content=str(output), tool_call_id=tool_call['id'])
            )
        
        # Return the tool outputs
        return {"messages": tool_outputs}
    else:
        # This should not happen in our loop, but as a fallback
        return {}

# --- 4. Define the "Edges" (The Agent's Flowchart) ---
def should_continue(state: AgentState):
    """This edge decides if the agent is done or needs to loop (to think more)."""
    last_message = state['messages'][-1]
    
    if last_message.tool_calls:
        # The brain has called a tool, so we must run the tool and then loop back
        return "call_tools"
    else:
        # The brain has *not* called a tool, meaning it has given its final answer
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