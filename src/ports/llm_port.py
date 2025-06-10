from abc import ABC, abstractmethod

class ILLMService(ABC):
    """Abstract Base Class for LLM services."""

    @abstractmethod
    def get_llm_response(self, prompt: str) -> str:
        """
        Generates a response from the LLM based on the given prompt.
        """
        pass