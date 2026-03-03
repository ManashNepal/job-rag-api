# Job Search RAG API

## Overview

This project is a Retrieval-Augmented Generation (RAG) API designed for job search.

It retrieves relevant job postings using hybrid search (semantic and keyword search), reranks them for better relevance and generates structured responses using a large language model.

The system separates retrieval, reranking and generation into clear stages to keep the pipeline modular and easy to understand.

---

## How It Works

1. A user sends a job-related query.
2. The query is converted into:
   - A dense embedding (semantic representation)
   - A sparse representation (BM25 keyword-based)
3. Pinecone performs hybrid search using both representations.
4. Retrieved results are reranked using a cross-encoder model.
5. The top results are passed to a large language model.
6. The API returns structured job recommendations.

---

## Tech Stack

- FastAPI
- Pinecone (Hybrid Search)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
- Groq LLM (`llama-3.3-70b-versatile`)
- Pydantic
- Docker

---

## Setup and Installation

### Prerequisites

- Docker installed
- Valid API keys:
  - GROQ_API_KEY
  - PINECONE_API_KEY
  - HF_TOKEN (Optional)

### Pull the Image

```bash
docker pull manashnepal/job-rag-api:latest
```

### Option A: Run Using `.env` File

Create a file named `.env`:

```bash
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
HF_TOKEN=your_hf_token
```

Run:

```bash
docker run -p 8000:8000 --env-file .env manashnepal/job-rag-api:latest
```

### Option B: Run Using Inline Environment Variables

For macOS/Linux:

```bash
docker run -p 8000:8000 \
-e GROQ_API_KEY=your_groq_api_key \
-e PINECONE_API_KEY=your_pinecone_api_key \
-e HF_TOKEN=your_hf_token \
manashnepal/job-rag-api:latest
```

For Windows PowerShell:

```powershell
docker run -p 8000:8000 `
-e GROQ_API_KEY=your_groq_api_key `
-e PINECONE_API_KEY=your_pinecone_api_key `
-e HF_TOKEN=your_hf_token `
manashnepal/job-rag-api:latest
```

---

## Accessing the API

After the container starts, open:

`http://localhost:8000/docs`

This provides the interactive Swagger UI for testing endpoints.

## Interacting with the API

### Base URL

http://localhost:8000

### Endpoint

`POST /search`

### Example Request Body

```json
{
  "query": "machine learning engineer remote"
}
```

### Example Response

```json
{
  "results": [
    {
      "title": "Machine Learning Engineer",
      "company": "Example Corp",
      "location": "Remote",
      "summary": "Role focused on building ML systems and data pipelines."
    },
    {
      "title": "Machine Learning Engineer",
      "company": "Example Corp",
      "location": "Remote",
      "summary": "Role focused on building ML systems and data pipelines."
    },
    {
      "title": "Machine Learning Engineer",
      "company": "Example Corp",
      "location": "Remote",
      "summary": "Role focused on building ML systems and data pipelines."
    }
  ]
}
```
