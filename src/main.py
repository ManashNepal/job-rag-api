from fastapi import FastAPI
import logging
from src.rag.router import router

# Configure logging once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

app = FastAPI(
    title="Job RAG API",
)

app.include_router(router)
