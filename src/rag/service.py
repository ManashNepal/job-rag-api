from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import time

from src.config import settings
from src.rag.schemas import JobSearchResponse
from src.rag.retriever import HybridRetriever
from src.rag.reranker import Reranker


DEFAULT_TOP_K = 3


class RAGService:
    """Service class implementing the RAG pipeline for job search."""

    def __init__(self):
        """Initialize vector store, retriever, LLM, and prompt template."""
        self.retriever = HybridRetriever(top_k=8)
        self.reranker = Reranker(top_k=DEFAULT_TOP_K)

        base_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=settings.groq_api_key,
        )

        self.llm = base_llm.with_structured_output(
            schema=JobSearchResponse, strict=True
        )

        self.prompt = ChatPromptTemplate.from_template(
            """
You are an AI job search assistant.

Use ONLY the provided job context to answer.

Return the most relevant jobs for the user query.

Rules:
- Maximum 5 jobs
- Do NOT invent information
- If nothing relevant, return an empty list
- Order results by relevance

Summary Guidelines:
- Write a clear and informative summary describing the overall role
- Start with a descriptive phrase such as:
  "The role involves", "This position focuses on", "The job centers on",
  "The position involves", or a similar natural variation
- Use only ONE opening phrase and avoid repeating similar phrases
- Include the main responsibilities and purpose of the job
- Focus on what the candidate will do and the type of work involved
- If company information appears first, prioritize role details instead
- Avoid generic company descriptions, workplace environment details, or marketing text
- Keep the summary concise (2-3 sentences maximum)

Context:
{context}

User Query:
{query}
"""
        )

    def query(self, user_query: str):
        """
        Execute the RAG pipeline:
        retrieve relevant jobs, build context, invoke LLM, and return results.
        """
        start = time.time()
        docs = self.retriever.search(user_query)
        # Rerank documents
        docs = self.reranker.rerank(user_query, docs)
        retrieval_time = time.time() - start

        context = self.build_context(docs)

        messages = self.prompt.format_messages(
            context=context,
            query=user_query,
        )

        start_llm = time.time()
        response = self.llm.invoke(messages)
        llm_time = time.time() - start_llm

        total_time = time.time() - start

        print(
            f"Retrieval: {retrieval_time:.2f}s | LLM: {llm_time:.2f}s | Total: {total_time:.2f}s"
        )

        return response

    def build_context(self, docs):
        """Construct a compact text context from retrieved documents to send to the LLM."""
        context_parts = []

        if not docs:
            return ""

        for doc in docs:
            metadata = doc.metadata or {}
            title = metadata.get("title", "N/A")
            company = metadata.get("company", "N/A")
            location = metadata.get("location", "N/A")

            # Extract only description portion
            content = doc.page_content

            if "Description:" in content:
                description = content.split("Description:", 1)[-1].strip()
            else:
                description = content

            description = description[:600]  # shorter gives faster response

            text = f"""
---
Job Title: {title}
Company: {company}
Location: {location}
Description: {description}
---
"""

            context_parts.append(text)

        return "\n\n".join(context_parts)
