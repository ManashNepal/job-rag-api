import logging
import time

import anthropic
from langchain_core.documents import Document

from src.config import settings
from src.rag.query_parser import QueryParser
from src.rag.reranker import Reranker
from src.rag.retriever import HybridRetriever
from src.rag.schemas import Job, JobSearchResponse, LLMJobMatches, QueryFilters

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You help people find jobs. You get a search query and numbered job postings \
([Job 1], [Job 2], ...), each an excerpt of the full posting.

Return the postings that are relevant to the query, most relevant first. Leave out postings \
that don't fit; return an empty list if none do. For each one:
- summary: 2-3 sentences on what the role involves (responsibilities, main technologies). \
Skip company marketing.
- match_reason: one sentence on how it matches the query, or where it falls short \
(e.g. "Senior role, but the query asks for junior").

Use only information from the postings."""


class RAGService:
    """Service class implementing the RAG pipeline for job search."""

    def __init__(self):
        self.retriever = HybridRetriever(top_k=settings.retriever_top_k)
        self.reranker = Reranker(top_k=settings.reranker_top_k)
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.query_parser = QueryParser(self.client)

    def query(self, user_query: str) -> JobSearchResponse:
        """Retrieve and rerank jobs, ask LLM which ones match, and build the response."""
        start = time.perf_counter()
        filters = self.query_parser.parse(user_query)
        filter_time = time.perf_counter() - start

        start_retrieval = time.perf_counter()
        docs, filters_relaxed = self.retrieve(user_query, filters)
        retrieval_time = time.perf_counter() - start_retrieval

        start_rerank = time.perf_counter()
        docs = self.reranker.rerank(user_query, docs)
        rerank_time = time.perf_counter() - start_rerank

        if not docs:
            return JobSearchResponse(
                filters=filters, filters_relaxed=filters_relaxed, results=[]
            )

        start_llm = time.perf_counter()
        matches = self.find_matches(user_query, docs)
        llm_time = time.perf_counter() - start_llm

        logger.info(
            "Filters: %.2fs | Retrieval: %.2fs | Rerank: %.2fs | LLM: %.2fs | Total: %.2fs",
            filter_time,
            retrieval_time,
            rerank_time,
            llm_time,
            time.perf_counter() - start,
        )

        return JobSearchResponse(
            filters=filters,
            filters_relaxed=filters_relaxed,
            results=self.build_results(matches, docs),
        )

    def retrieve(
        self, user_query: str, filters: QueryFilters
    ) -> tuple[list[Document], bool]:
        """Search with the filters; if too few jobs match, search again without them.

        Returns the documents and whether the filters were dropped.
        """
        if filters.is_empty():
            return self.retriever.search(user_query), False

        docs = self.retriever.search(user_query, filters)
        if len(docs) >= settings.min_filtered_results:
            return docs, False

        logger.info(
            "Only %d jobs matched filters %s, searching without filters",
            len(docs),
            filters.model_dump(exclude_none=True),
        )
        return self.retriever.search(user_query), True

    def find_matches(self, user_query: str, docs: list[Document]) -> LLMJobMatches:
        """Ask LLM which of the numbered jobs match the query."""
        response = self.client.beta.messages.parse(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"{self.build_context(docs)}\n\nSearch query: {user_query}",
                }
            ],
            output_format=LLMJobMatches,
            output_config={"effort": settings.llm_effort},
            # If the model declines, retry on the fallback model
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

        if response.stop_reason == "refusal":
            logger.warning("LLM declined the request: %s", response.stop_details)
            return LLMJobMatches(matches=[])
        if response.stop_reason == "max_tokens":
            raise RuntimeError("LLM response was cut off (max_tokens reached)")

        return response.parsed_output

    def build_context(self, docs: list[Document]) -> str:
        """Number each job's best chunk so LLM can refer to it as [Job N]."""
        return "\n\n".join(
            f"[Job {number}]\n{doc.page_content}"
            for number, doc in enumerate(docs, start=1)
        )

    def build_results(self, matches: LLMJobMatches, docs: list[Document]) -> list[Job]:
        """Combine LLM's summaries with the job data from the index."""
        results, seen = [], set()

        for match in matches.matches:
            # Ignore numbers that don't point to a job we sent or repeats
            if not 1 <= match.job_number <= len(docs) or match.job_number in seen:
                continue
            seen.add(match.job_number)

            metadata = docs[match.job_number - 1].metadata
            results.append(
                Job(
                    title=metadata["title"],
                    company=metadata["company"],
                    location=metadata["location"],
                    country=metadata["country"],
                    work_type=metadata["work_type"],
                    job_level=metadata["job_level"],
                    seniority=metadata.get("seniority", "unspecified"),
                    job_link=metadata["job_link"],
                    summary=match.summary,
                    match_reason=match.match_reason,
                    score=metadata["rerank_score"],
                )
            )

        return results
