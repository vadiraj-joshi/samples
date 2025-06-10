import time
from src.ports.llm_port import ILLMService

class DummyLLMAdapter(ILLMService):
    """
    A dummy LLM adapter for PoC purposes.
    It returns a generic response and simulates a delay.
    """
    def get_llm_response(self, prompt: str) -> str:
        # Simulate some processing time
        time.sleep(0.1)
        if "summarize" in prompt.lower():
            return f"This is a dummy summary of: '{prompt[:50]}...'"
        elif "translate" in prompt.lower():
            return f"This is a dummy translation of: '{prompt[:50]}...'"
        else:
            return f"Dummy LLM response to: '{prompt}'"