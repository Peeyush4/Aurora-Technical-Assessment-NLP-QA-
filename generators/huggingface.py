from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from .base import BaseGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"

class HuggingFaceGenerator(BaseGenerator):
    def __init__(self, model_name="google/flan-t5-base"):
        try:
            print(f"Initializing local T5 model: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.eval()
            print("Hugging Face Generator initialized.")
        except Exception as e:
            print(f"Error initializing Hugging Face model: {e}")
            self.model = None
            self.tokenizer = None

    def generate_with_cpu(self, prompt: str) -> str:
        self.model.to('cpu')
        enc = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(**enc, max_new_tokens=100)
        output_ids = outputs[0].cpu()
        answer = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return answer
    
    def generate_with_gpu(self, prompt: str, device: str) -> str:
        # Tokenize the input
        enc = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        self.model.to(device)
        # # Move each tensor to the same device as the model
        # model_device = next(self.model.parameters()).device
        # moved_inputs = {}
        # for k, v in enc.items():
        #     if hasattr(v, 'to'):
        #         moved_inputs[k] = v.to(model_device)
        #     else:
        #         moved_inputs[k] = v

        # # Diagnostic logging: show devices
        # try:
        #     devices = {k: (v.device if hasattr(v, 'device') else None) for k, v in moved_inputs.items()}
        # except Exception:
        #     devices = {k: str(type(v)) for k, v in moved_inputs.items()}
        # print(f"Model device: {model_device}; input tensor devices: {devices}")

        # Use no_grad for inference and generate on the same device
        with torch.no_grad():
            outputs = self.model.generate(**enc.to(device), max_new_tokens=100)

        # Move output tensor to CPU before decoding
        output_ids = outputs[0].cpu()
        answer = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return answer
    
    def generate(self, prompt: str, device: str = device) -> str:
        if self.model is None or self.tokenizer is None:
            return "Error: Hugging Face model is not initialized."
        if device != "cpu":
            try:
                return self.generate_with_gpu(prompt, device)
            except Exception as e:
                print(f"Error calling local T5 model: {e}")
                # If CUDA-related device mismatch, try a CPU fallback
                try:
                    if str(device).startswith('cuda'):
                        print("Attempting CPU fallback for generation...")
                        return self.generate_with_cpu(prompt)
                except Exception as e2:
                    print(f"CPU fallback also failed: {e2}")
                return "Error: Could not generate answer from local model."
        else:
            return self.generate_with_cpu(prompt)
        
