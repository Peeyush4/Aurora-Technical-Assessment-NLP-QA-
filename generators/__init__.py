import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Read the "switch" from your .env file
MODEL_TYPE = os.getenv("GENERATOR_MODEL", "litellm").lower()

if MODEL_TYPE == "gemini":
    print("Using Gemini Generator.")
    from .gemini import GeminiGenerator
    generator = GeminiGenerator()
elif MODEL_TYPE == "huggingface":
    print("Using Hugging Face Generator.")
    from .huggingface import HuggingFaceGenerator
    generator = HuggingFaceGenerator()
elif MODEL_TYPE == "litellm":
    print("Using LiteLLM Generator.")
    from .litellm import LiteLLMGenerator
    MODEL_NAME = os.getenv(
        "LITELLM_MODEL_NAME", 
        "ollama/llama3.1:8b"
    )
    generator = LiteLLMGenerator(model_name=MODEL_NAME)
else:
    raise ValueError(f"Unknown GENERATOR_MODEL type in .env: {MODEL_TYPE}")