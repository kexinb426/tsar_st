# src/models/gpt.py
import os
from openai import OpenAI
from .base_model import SimplificationModel

class GPTModel(SimplificationModel):
    """
    A wrapper for OpenAI's GPT models.
    """
    def __init__(self, api_model_name: str, temperature: float = 0.7):
        """
        Initializes the GPT model wrapper.
        
        Args:
            api_model_name: The exact model ID for the OpenAI API call.
            temperature: The sampling temperature to use.
        """
        try:
            self.client = OpenAI()
        except Exception as e:
            raise Exception("Failed to initialize OpenAI client. Is your OPENAI_API_KEY set?") from e
        
        self.model_name = api_model_name
        self.temperature = temperature
        print(f"✅ GPTModel initialized with API model name: {self.model_name}, Temperature: {self.temperature}")

    def simplify(self, prompt: str) -> tuple[str, str]:
        """
        Sends the prompt to the specified GPT model, parses the output for Chain-of-Thought,
        and returns both the raw output and the final simplified text.

        Returns:
            A tuple containing (raw_model_output, final_simplified_text).
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in text simplification."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
            )
            raw_output = response.choices[0].message.content.strip()

            # --- Parse the output for Chain-of-Thought ---
            cot_marker = "Simplified Document:"
            
            if cot_marker in raw_output:
                # Use rsplit to split from the right, ensuring we get the final answer.
                parts = raw_output.rsplit(cot_marker, 1)
                final_answer = parts[1].strip()
                return raw_output, final_answer
            else:
                # If the marker isn't found, the raw output is the final answer.
                return raw_output, raw_output

        except Exception as e:
            print(f"❌ An error occurred while calling the OpenAI API: {e}")
            return "", ""