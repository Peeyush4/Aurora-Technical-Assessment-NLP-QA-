import os
from .base import BaseGenerator
from litellm import completion
from dotenv import load_dotenv
import ollama
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
        self.provider = self.model_name.split("/")[0]
        print(self.model_name, self.provider)
        print(f"Provider detected: {self.provider}")
        if self.provider == "ollama":
            try:
                print("Checking for model presence via Ollama...")
                ollama.show(self.model_name.split("/")[-1])
                print(f"Model {self.model_name} is present locally.")
            except: 
                print(f"Model {self.model_name} not found locally. Pulling...")
                ollama.pull(self.model_name.split("/")[-1])
                print(f"Model {self.model_name} pulled successfully.")

        # Set the HUGGINGFACE_API_KEY in the environment for litellm
        # This is the correct way to auth for HF models
        if "huggingface/" in model_name and not os.getenv("HUGGINGFACE_API_KEY"):
             print("Warning: HUGGINGFACE_API_KEY not set in .env for Hugging Face model.")


    def generate(self, prompt: str) -> str:
        """
        Takes a full prompt and returns a string answer.
        """
        try:
            # if self.model_name in ["mistral", "llama3"]: # Add any local models here
            #     self.model_name = f"ollama/{self.model_name}"
            response = ""
            if self.provider == "huggingface":    
                response = completion(
                    model=self.model_name,
                    messages=[
                        {"content": prompt, "role": "user"}
                    ],
                    api_key=os.getenv("HUGGINGFACE_API_KEY")
                )
            else:
                response = completion(
                    model=self.model_name,
                    messages=[
                        {"content": prompt, "role": "user"}
                    ],
                )
            
            content = (
                response.choices[0].message.content
                or "<no response>"
            )
            return content

        except Exception as e:
            print(f"--- !!! LITELLM ERROR !!! ---")
            print(f"Error calling model {self.model_name}: {e} with provider {self.provider}")
            print("If using Ollama, is 'ollama serve' running?")
            print("If using HuggingFace, is the model public or is your key valid?")
            print(e)
            print("-----------------------------------")
            return "Error: Could not generate answer."