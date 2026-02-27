from fastapi import APIRouter, HTTPException

from src.rag.schemas import JobQueryRequest, JobQueryResponse, JobResult
from src.rag.service import RAGService


router = APIRouter()
rag_service = RAGService()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/search", response_model=JobQueryResponse)
def search_jobs(request: JobQueryRequest):

    try:
        jobs = rag_service.query(request.query)

        results = [JobResult(**job) for job in jobs]

        return JobQueryResponse(
            query=request.query,
            results=results,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
