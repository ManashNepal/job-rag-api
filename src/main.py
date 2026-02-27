from fastapi import FastAPI
from src.rag.router import router


app = FastAPI(
    title="Job RAG API",
)

app.include_router(router)
