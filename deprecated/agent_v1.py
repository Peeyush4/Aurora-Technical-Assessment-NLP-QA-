from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from tools import all_tools

# --- 1. Define the Agent's "Memory" (State) ---
class AgentState(TypedDict):
    # The list of messages in our chat
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# --- 2. Define the Agent's "Brain" (The LLM) ---
# This prompt defines the "personality" and "reasoning rules"
# This is where we solve the "contradiction" and "alias" problems
AGENT_SYSTEM_PROMPT = """
You are an expert assistant. Your job is to answer questions about a user's profile.
You have two tools: `find_user_names` and `search_messages`.

**Your Reasoning Process:**
1.  **ALWAYS** use the `find_user_names` tool *first* to identify which user the question is about.
2.  Once you have the user's *full, correct name*, **ALWAYS** use the `search_messages` tool to find relevant messages from that user's history.
3.  Look at the list of messages you retrieved. This is your *only* source of truth.
4.  Analyze these messages to answer the question. You MUST follow these rules:
    * **Contradiction Rule:** If you find contradictory information (e.g., "window seat" and "aisle seat"), the message with the **latest timestamp** is the correct one.
    * **Reasoning Rule:** You must reason about the messages. If a "thank you" message (e.g., "concert package was great") follows a "request" message (e.g., "son's graduation event"), they are connected.
    * **"No Car" Rule:** If you search for "owns a car" and only find "car service," the answer is "0 cars owned."
5.  Formulate a final, helpful answer for the user.
"""

# The LLM "Brain"
# We bind the tools so Mistral knows it can call them
llm = ChatOllama(model="mistral", temperature=0)
llm_with_tools = llm.bind_tools(all_tools)

# --- 3. Define the "Nodes" (The Agent's Actions) ---

# This node is the "brain"
def call_model_node(state: AgentState):
    """The primary node that calls the LLM (Mistral) to decide what to do."""
    print("--- Node: call_model ---")
    
    # Get all messages from the state
    messages = state['messages']
    
    # Add the system prompt *only* if it's the first message
    if len(messages) == 1:
        messages = [
            HumanMessage(content=AGENT_SYSTEM_PROMPT),
            messages[0]
        ]
        
    # Call the LLM. Mistral will decide to call a tool or answer.
    response = llm_with_tools.invoke(messages)
    
    # The agent's response is added to the state
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
            
            # Find the correct tool from our list
            tool_to_call = {t.name: t for t in all_tools}[tool_name]
            
            # Run the tool
            print(f"  -> Calling Tool: {tool_name}({tool_call['args']})")
            output = tool_to_call.invoke(tool_call['args'])
            
            # Format the output as a ToolMessage
            tool_outputs.append(
                ToolMessage(content=str(output), tool_call_id=tool_call['id'])
            )
        
        # Add the tool's answer back to the state
        return {"messages": tool_outputs}
    else:
        # This shouldn't happen, but just in case
        return {}


# --- 4. Define the "Edges" (The Agent's Flowchart) ---

def should_continue(state: AgentState):
    """This edge decides if the agent is done or needs to loop."""
    last_message = state['messages'][-1]
    
    # If the model called a tool, we must loop back to the model
    if last_message.tool_calls:
        return "continue"
    else:
        # If the model just sent a plain text answer, we're done
        return "end"

# --- 5. Build the Graph ---

print("Building LangGraph agent...")
# Initialize the graph and set the "memory"
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("agent", call_model_node) # The "brain"
workflow.add_node("tools", call_tool_node) # The "hands"

# Add the edges (the flowchart)
workflow.set_entry_point("agent") # Start with the brain

# This is the "loop"
workflow.add_conditional_edges(
    "agent", # After the brain runs...
    should_continue, # ...check if it's done or not
    {
        "continue": "tools", # If it called a tool, go to the "tools" node
        "end": END           # If it's done, end the process
    }
)

# This edge sends the tool results back to the brain
workflow.add_edge("tools", "agent")

# We're done! Compile the graph into a runnable "app"
# We add a memory saver so it can remember conversations (optional but cool)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
print("LangGraph Agent compiled successfully.")