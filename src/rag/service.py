from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

import re

from src.config import settings
from src.indexing.embedder import get_embedding_model


DEFAULT_TOP_K = 5


class RAGService:
    def __init__(self):

        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)

        self.embedding_model = get_embedding_model()

        self.vectorstore = PineconeVectorStore(
            index=index,
            embedding=self.embedding_model,
        )

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": DEFAULT_TOP_K}
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=0.2,
            google_api_key=settings.google_api_key,
        )

        self.prompt = PromptTemplate(
            template="""
You are a professional AI job search assistant.

STRICT RULES:
- Use ONLY the provided job context.
- Do NOT invent jobs.
- If no relevant jobs exist, say: No relevant jobs found.
- Return up to 5 jobs.

FORMAT:

Job Title: ...
Company: ...
Location: ...
Summary: ...

CONTEXT:
{context}

USER QUERY:
{query}
""",
            input_variables=["context", "query"],
        )

    def query(self, user_query: str):

        docs = self.retriever.invoke(user_query)

        context = self.build_context(docs)

        formatted_prompt = self.prompt.format(
            context=context,
            query=user_query,
        )

        response = self.llm.invoke(formatted_prompt)

        text = self.extract_text(response)

        return self.parse_jobs(text)

    def build_context(self, docs):

        context_parts = []
        total_chars = 0
        MAX_CHARS = 4000

        for doc in docs:
            metadata = doc.metadata or {}

            text = f"""
Job Title: {metadata.get("title", "N/A")}
Company: {metadata.get("company", "N/A")}
Location: {metadata.get("location", "N/A")}

Description:
{doc.page_content}
"""

            if total_chars + len(text) > MAX_CHARS:
                break

            context_parts.append(text)
            total_chars += len(text)

        return "\n\n".join(context_parts)

    def extract_text(self, response) -> str:

        if hasattr(response, "content"):
            content = response.content

            if isinstance(content, list):
                texts = []

                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        texts.append(block["text"])

                return "\n".join(texts)

            return content

        return str(response)

    def parse_jobs(self, text: str):

        jobs = []

        pattern = re.compile(
            r"Job Title:\s*(.*?)\n.*?Company:\s*(.*?)\n.*?Location:\s*(.*?)\n.*?Summary:\s*(.*?)(?:\n\n|$)",
            re.DOTALL | re.IGNORECASE,
        )

        matches = pattern.findall(text)

        for match in matches:
            jobs.append(
                {
                    "title": match[0].strip(),
                    "company": match[1].strip(),
                    "location": match[2].strip(),
                    "summary": match[3].strip(),
                }
            )

        return jobs
