import os
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# We import all the tools for the "brain" to use
from tools import all_tools
from langchain_core.utils.function_calling import convert_to_openai_tool
import json

# --- 1. Define the Agent's "Memory" (State) ---
# We go back to the simple, standard, powerful state
class AgentState(TypedDict):
    # This holds the full chat history
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# --- 2. Define the Agent's "Brain" (The LLM) ---

# This is the agent's instruction set.
# Primary behavior: prefer existing data; call tools only when necessary.
AGENT_SYSTEM_PROMPT = """
VARIANT 5 (HYBRID: EXTRACT-FIRST, TOOL-IF-MISSING):
You are an expert assistant. Primary rule: attempt to answer from existing messages or RAG hits first by extracting structured values (phones, credit card-like numbers, preferences). If you can extract a reliable value, return a concise answer (single best value) without calling tools.

If you cannot find the information in the available messages, call the appropriate retrieval tool using a single provider-style function call (populate the tool_calls field with name + parameters) and nothing else.

When returning phone numbers, prefer the latest timestamped match and format as digits-only. Be deterministic (temperature=0).
"""

# The LLM "Brain"
# Create the ChatOllama model with deterministic temperature; pass tools
# at invoke-time as provider-ready specs to avoid validation errors.
llm = ChatOllama(model="llama3.1:8b", temperature=0)
llm_with_tools = llm.bind_tools(all_tools)
# Prepare provider-ready tool specs for Ollama (convert LangChain tools)
tools_for_request = []
for t in all_tools:
    try:
        tools_for_request.append(convert_to_openai_tool(t))
    except Exception:
        # If conversion fails for a tool, skip it and continue; the bound
        # `llm_with_tools` still knows about tools, but passing explicit
        # converted specs helps the Ollama client perform function-calling.
        continue

# --- 3. Define the "Nodes" (The Agent's Actions) ---

# This node is the "brain"
def call_model_node(state: AgentState):
    """The primary node that calls the LLM (Mistral) to decide what to do."""
    print("--- Node: call_model (Agent Brain) ---")
    
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + state['messages']
    # Primary invoke: provider-native function-calling when available
    response = llm_with_tools.invoke(messages, format="json")
    print(f"  -> LLM Response: {repr(response)}")
    print("tool_calls attr:", getattr(response, "tool_calls", None))
    print("content/text:", getattr(response, "content", getattr(response, "text", None)))
    print("invalid_tool_calls:", getattr(response, "invalid_tool_calls", None))
    print("dir(response):", [n for n in dir(response) if not n.startswith('_')])
    # Return response directly. If it contains structured tool_calls the run_tools node will execute them.
    return {"messages": [response]}

# This node runs the tools
def call_tool_node(state: AgentState):
    """This node checks for tool calls and executes them."""
    print("--- Node: call_tool ---")
    
    last_message = state['messages'][-1]
    
    # If the last message is a tool call, run it
    if last_message.tool_calls:
        tool_outputs = []
        tool_map = {t.name: t for t in all_tools}
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            if tool_name not in tool_map:
                print(f"  -> Unknown tool requested: {tool_name}; skipping.")
                tool_outputs.append(
                    ToolMessage(content=f"Error: unknown tool '{tool_name}'", 
                                tool_call_id=tool_call.get('id'))
                )
                continue

            tool_to_call = tool_map[tool_name]

            print(f"  -> Calling Tool: {tool_name}({tool_call['args']})")
            # Run the tool and keep the raw Python output if possible
            try:
                output = tool_to_call.invoke(tool_call['args'])
            except Exception as e:
                output = [f"Error invoking tool: {e}"]

            # Default: string-serialize the output for downstream LLM consumption
            tool_outputs.append(
                ToolMessage(content=json.dumps(output, default=str), tool_call_id=tool_call['id'])
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