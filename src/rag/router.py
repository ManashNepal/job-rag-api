from fastapi import APIRouter, HTTPException

from src.rag.schemas import JobQueryRequest, JobSearchResponse
from src.rag.service import RAGService


router = APIRouter()
rag_service = RAGService()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/search", response_model=JobSearchResponse)
async def search_jobs(request: JobQueryRequest):

    try:
        response = rag_service.query(request.query)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
