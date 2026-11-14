import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config import settings
from src.rag.router import router
from src.rag.service import RAGService

# Configure logging once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# Set before any model is downloaded
if settings.hf_token:
    os.environ["HF_TOKEN"] = settings.hf_token


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline once at startup."""
    app.state.rag_service = RAGService()
    yield


app = FastAPI(
    title="Job RAG API",
    lifespan=lifespan,
)

app.include_router(router)
