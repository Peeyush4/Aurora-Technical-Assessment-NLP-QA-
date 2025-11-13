import os
from .base import BaseGenerator
from litellm import completion
from dotenv import load_dotenv

# Load the .env file to get API keys
load_dotenv()

class LiteLLMGenerator(BaseGenerator):
    """
    A universal generator using litellm to call any model.
    Handles Ollama, HuggingFace, Gemini, etc.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        print(f"LiteLLM Generator initialized. Using model: {self.model_name}")
        
        # Set the HUGGINGFACE_API_KEY in the environment for litellm
        # This is the correct way to auth for HF models
        if "huggingface/" in model_name and not os.getenv("HUGGINGFACE_API_KEY"):
             print("Warning: HUGGINGFACE_API_KEY not set in .env for Hugging Face model.")


    def generate(self, prompt: str) -> str:
        """
        Takes a full prompt and returns a string answer.
        """
        try:
            # We must prefix ollama models so litellm knows where to find them
            model_to_call = self.model_name
            # if self.model_name in ["mistral", "llama3"]: # Add any local models here
            #     model_to_call = f"ollama/{self.model_name}"

            response = completion(
                model=model_to_call,
                messages=[
                    {
                        "content": prompt,
                        "role": "user"
                    }
                ],
                # No api_key needed here, litellm finds it
                # from the environment variables (like HUGGINGFACE_API_KEY)
            )
            
            content = (
                response.choices[0].message.content
                or "<no response>"
            )
            return content

        except Exception as e:
            print(f"--- !!! LITELLM ERROR !!! ---")
            print(f"Error calling model {self.model_name}: {e}")
            print("If using Ollama, is 'ollama serve' running?")
            print("If using HuggingFace, is the model public or is your key valid?")
            print("-----------------------------------")
            return "Error: Could not generate answer."