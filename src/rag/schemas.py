from pydantic import BaseModel, Field


# Internal Claude response
class JobMatch(BaseModel):
    job_number: int = Field(
        description="Number of the job in the context, e.g. 2 for [Job 2]"
    )
    summary: str = Field(description="2-3 sentences on what the role involves")
    match_reason: str = Field(description="One sentence on how the job fits the query")


class LLMJobMatches(BaseModel):
    matches: list[JobMatch] = Field(description="Relevant jobs, most relevant first")


# API response
class Job(BaseModel):
    # From the job's data
    title: str
    company: str
    location: str
    country: str
    work_type: str
    job_level: str
    job_link: str
    # Written by Claude
    summary: str
    match_reason: str
    # Reranker relevance score
    score: float


class JobQueryRequest(BaseModel):
    query: str = Field(description="User job search query")


class JobSearchResponse(BaseModel):
    results: list[Job] = Field(description="Relevant jobs, most relevant first")
