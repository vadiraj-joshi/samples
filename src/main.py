from fastapi import FastAPI
import uvicorn

# Import your router
from src.routers.evaluation_router import router as evaluation_router

app = FastAPI(
    title="LLM Evaluation PoC API",
    description="A simplified PoC for LLM evaluation using Hexagonal Architecture.",
    version="0.1.0",
)

# Include your routers
app.include_router(evaluation_router)

@app.get("/")
async def root():
    return {"message": "LLM Evaluation PoC API is running. Go to /docs for API documentation."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)