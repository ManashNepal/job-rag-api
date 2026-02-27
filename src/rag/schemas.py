from pydantic import BaseModel, Field
from typing import List


class JobResult(BaseModel):
    title: str
    company: str
    location: str
    summary: str


class JobQueryRequest(BaseModel):
    query: str = Field(..., description="User search query")


class JobQueryResponse(BaseModel):
    query: str
    results: List[JobResult]
