import logging

import anthropic
from fastapi import APIRouter, HTTPException, Request
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.config import settings
from src.rag.schemas import JobQueryRequest, JobSearchResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# LLM errors worth retrying later
LLM_UNAVAILABLE_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
    anthropic.ServiceUnavailableError,
)

QDRANT_ERRORS = (ResponseHandlingException, UnexpectedResponse)


@router.get("/health")
def health(request: Request):
    """Report whether the API can reach Qdrant and the job collection exists."""
    client = request.app.state.rag_service.retriever.client
    try:
        collection_ready = client.collection_exists(settings.qdrant_collection)
    except QDRANT_ERRORS:
        logger.exception("Health check: Qdrant unreachable")
        raise HTTPException(
            status_code=503, detail={"status": "unavailable", "qdrant": "unreachable"}
        )

    if not collection_ready:
        raise HTTPException(
            status_code=503,
            detail={"status": "unavailable", "qdrant": "collection missing"},
        )
    return {"status": "ok", "qdrant": "ok"}


@router.post("/search", response_model=JobSearchResponse)
def search_jobs(payload: JobQueryRequest, request: Request) -> JobSearchResponse:
    service = request.app.state.rag_service
    try:
        return service.query(payload.query)
    except LLM_UNAVAILABLE_ERRORS:
        logger.exception("LLM request failed")
        raise HTTPException(
            status_code=503,
            detail="The language model service is unavailable, please try again.",
        )
    except QDRANT_ERRORS:
        logger.exception("Qdrant request failed")
        raise HTTPException(
            status_code=503,
            detail="The search index is unavailable, please try again.",
        )
    except Exception:
        logger.exception("Search failed")
        raise HTTPException(
            status_code=500, detail="Something went wrong while searching."
        )
