# src/models/base_model.py
from abc import ABC, abstractmethod

class SimplificationModel(ABC):
    """
    An abstract base class for all simplification models.
    """
    @abstractmethod
    def simplify(self, prompt: str) -> tuple[str, str]:
        """
        Takes a fully-formed prompt and returns the model's response.

        Args:
            prompt: The final prompt string to be sent to the model.

        Returns:
            A tuple containing:
            (raw_model_output: str, final_simplified_text: str)
        """
        pass