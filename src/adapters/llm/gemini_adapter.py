# src/llm_evaluation/adapters/llm/gemini_adapter.py
from src.ports.llm_port import ILLMService
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.logging.logger import logger

class GeminiLLMAdapter(ILLMService):
    """
    Driven Adapter: Implements the ILLMService port using the Gemini LLM.
    This adapter translates between the domain's needs and the external LLM service.
    """
    def __init__(self, llm_client: LLMClient):
        self._llm_client = llm_client
        logger.info("GeminiLLMAdapter initialized.")

    async def get_llm_response(self, prompt: str) -> str:
        """Sends a prompt to the LLM via the client and returns its response."""
        logger.debug(f"Sending prompt to LLM: {prompt[:100]}...")
        return await self._llm_client.generate_content(prompt)