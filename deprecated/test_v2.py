# scripts/ollama_debug.py
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool, convert_to_json_schema
from agent import app  # optional if you want agent pieces
from tools import all_tools
from langchain_ollama.chat_models import ChatOllama
import json, pprint

# Create a ChatOllama instance matching your agent config
llm = ChatOllama(model="mistral", temperature=0.0)  # deterministic for debugging

# Build messages
AGENT_SYSTEM_PROMPT = """You are an expert assistant. If you need to call a tool, return a structured tool call."""
messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT), 
            HumanMessage(content="Get me the phone number of Thiago Monteiro?")]

# Convert `all_tools` to provider function specs
tools_for_request = []
for t in all_tools:
    try:
        spec = convert_to_openai_tool(t)
        tools_for_request.append(spec)
    except Exception as e:
        print("convert_to_openai_tool failed for", getattr(t, "name", repr(t)), ":", e)

print("TOOLS FOR REQUEST:")
pprint.pprint(tools_for_request)

# Optionally also build JSON schema for a specific tool's args (example: first tool if it has args schema)
try:
    json_schema = None
    if tools_for_request:
        # convert_to_json_schema expects a Pydantic model or similar; attempt for tools that have schema
        json_schema = convert_to_json_schema(tools_for_request[0])
        print("SAMPLE JSON SCHEMA for tool 0:", json.dumps(json_schema, indent=2))
except Exception as e:
    print("convert_to_json_schema error:", e)

# Invoke model with explicit tools and format hint (try format='json' and format=None)
for fmt in (None, "json"):
    print("\n--- Invoke with format =", fmt, "---\n")
    try:
        resp = llm.invoke(messages, tools=tools_for_request, format=fmt)
    except Exception as e:
        print("llm.invoke raised:", type(e).__name__, e)
        continue

    # Print top-level response attributes
    print("RESP repr:", repr(resp))
    print("TOOL CALLS:", getattr(resp, "tool_calls", None))
    print("INVALID TOOL CALLS:", getattr(resp, "invalid_tool_calls", None))
    print("CONTENT (first 1000 chars):", (getattr(resp, "content", "") or "")[:1000])

    # The low-level server message may be in response.response_metadata['message']
    rm = getattr(resp, "response_metadata", None)
    if rm and isinstance(rm, dict):
        print("\\n-- response_metadata keys:", list(rm.keys()))
        if "message" in rm:
            print("-- raw server message (message):")
            pprint.pprint(rm["message"])
        if "message" in rm and isinstance(rm["message"], dict):
            # Print raw message.tool_calls if present
            print("-- message.tool_calls:", rm["message"].get("tool_calls"))
    print("\n----\n")