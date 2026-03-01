from pydantic import BaseModel, Field
from typing import List


class Job(BaseModel):
    title: str = Field(description="Job title or role name")
    company: str = Field(description="Company offering the job")
    location: str = Field(
        description="Job location including the city and country if available"
    )
    summary: str = Field(
        description="Short 2-3 sentence summary of the job responsibilities"
    )


class JobQueryRequest(BaseModel):
    query: str = Field(description="User job search query")


class JobSearchResponse(BaseModel):
    results: List[Job] = Field(
        description="List of relevant jobs matching the user query"
    )
