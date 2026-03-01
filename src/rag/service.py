from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import time

from src.config import settings
from src.indexing.vectorstore import get_vectorstore
from src.rag.schemas import JobSearchResponse


DEFAULT_TOP_K = 3


class RAGService:
    """Service class implementing the RAG pipeline for job search."""

    def __init__(self):
        """Initialize vector store, retriever, LLM, and prompt template."""
        self.vectorstore = get_vectorstore()

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": DEFAULT_TOP_K}
        )

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
- Focus summaries on job responsibilities and required skills
- Avoid generic company descriptions or marketing text

Context:
{context}

User Query:
{query}
"""
        )

    def query(self, user_query: str):
        start = time.time()
        docs = self.retriever.invoke(user_query)
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

            description = description[:400]  # shorter gives faster response

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
