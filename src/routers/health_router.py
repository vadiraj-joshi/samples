# src/llm_evaluation/routers/health_router.py
from fastapi import APIRouter, status

router = APIRouter(tags=["Health"])

@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Returns a simple health check status."""
    return {"status": "ok", "message": "LLM Evaluation Framework is up and running!"}
