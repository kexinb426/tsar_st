# src/models/gemma.py
import torch
from transformers import pipeline
from .base_model import SimplificationModel

class GemmaModel(SimplificationModel):
    """
    A wrapper for Gemma models via the Hugging Face transformers library.
    """
    def __init__(self, api_model_name: str, temperature: float = 0.7):
        """
        Initializes the Gemma model wrapper.
        
        Args:
            api_model_name: The Hugging Face Hub model ID (e.g., "google/gemma-2-9b-it").
            temperature: The sampling temperature to use.
        """
        try:
            # Initialize the text generation pipeline
            # device_map="auto" will automatically use available GPUs.
            self.pipeline = pipeline(
                "text-generation",
                model=api_model_name,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
        except Exception as e:
            raise Exception(f"Failed to initialize Hugging Face pipeline for model {api_model_name}. Is the model name correct and do you have access?") from e
        
        self.model_name = api_model_name
        self.temperature = temperature
        print(f"✅ GemmaModel initialized with HF model name: {self.model_name}, Temperature: {self.temperature}")

    def simplify(self, prompt: str) -> tuple[str, str]:
        """
        Sends the prompt to the loaded Gemma model and returns the response.

        Returns:
            A tuple containing (raw_model_output, final_simplified_text).
        """
        try:
            # Gemma instruct models use a specific chat template.
            # The pipeline handles this automatically when we provide a list of messages.
            messages = [
                {"role": "user", "content": prompt},
            ]

            # The pipeline returns a list of conversations. We take the first one.
            # The output contains the full conversation, including the user's prompt.
            outputs = self.pipeline(
                messages,
                max_new_tokens=4096, # Set a limit on the generated tokens
                do_sample=True,
                temperature=self.temperature,
            )
            
            # The 'generated_text' contains the full conversation history.
            # The last message in the list is the model's reply.
            raw_output = outputs[0]["generated_text"][-1]['content'].strip()

            # --- Parse for Chain-of-Thought ---
            cot_marker = "Simplified Document:"
            if cot_marker in raw_output:
                parts = raw_output.rsplit(cot_marker, 1)
                final_answer = parts[1].strip()
                return raw_output, final_answer
            else:
                return raw_output, raw_output

        except Exception as e:
            print(f"❌ An error occurred while running the Gemma pipeline: {e}")
            return "", ""