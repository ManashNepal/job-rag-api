from typing import Literal

from pydantic import BaseModel, Field

from src.config import settings
from src.indexing.preprocess import SENIORITY_LEVELS

# Allowed filter values
WorkType = Literal["remote", "hybrid", "onsite"]
Country = Literal[tuple(settings.countries)]
Seniority = Literal[SENIORITY_LEVELS]


# Filters extracted from the query
class QueryFilters(BaseModel):
    work_type: WorkType | None = Field(
        default=None, description="Remote, hybrid or onsite, if the query asks for one"
    )
    country: Country | None = Field(  # type: ignore
        default=None,
        description="Country the jobs should be in, if the query names one",
    )
    seniority: Seniority | None = Field(  # type: ignore
        default=None, description="Experience level, if the query asks for one"
    )

    def is_empty(self) -> bool:
        return (
            self.work_type is None and self.country is None and self.seniority is None
        )


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
    seniority: str
    job_link: str
    # Written by Claude
    summary: str
    match_reason: str
    # Reranker relevance score
    score: float


class JobQueryRequest(BaseModel):
    query: str = Field(description="User job search query")


class JobSearchResponse(BaseModel):
    filters: QueryFilters = Field(
        default_factory=QueryFilters, description="Filters understood from the query"
    )
    filters_relaxed: bool = Field(
        default=False,
        description="True if too few jobs matched and filters were dropped",
    )
    results: list[Job] = Field(description="Relevant jobs, most relevant first")
