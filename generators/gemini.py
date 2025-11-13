import os
import google.generativeai as genai
from .base import BaseGenerator

generation_config = {
  "temperature": 0.2,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 1024,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

class GeminiGenerator(BaseGenerator):
    def __init__(self):
        try:
            API_KEY = os.getenv("GOOGLE_API_KEY")
            if not API_KEY:
                raise ValueError("GOOGLE_API_KEY not found in .env file")
            genai.configure(api_key=API_KEY)
            self.model = genai.GenerativeModel(
                model_name=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            print("Gemini Generator initialized.")
        except Exception as e:
            print(f"Error initializing Gemini: {e}")
            self.model = None

    def generate(self, prompt: str) -> str:
        if self.model is None:
            return "Error: Gemini model is not initialized."
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return f"Error: Could not generate answer from Gemini. (Reason: {e})"