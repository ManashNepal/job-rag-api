from fastapi import FastAPI
import logging
from src.rag.router import router
from src.config import settings
import os

# Configure logging once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

if settings.hf_token:
    os.environ["HF_TOKEN"] = settings.hf_token

app = FastAPI(
    title="Job RAG API",
)

app.include_router(router)
