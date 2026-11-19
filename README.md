# Job Search RAG API

Search ~7,800 software engineering job postings in plain English, for example *"remote junior react jobs in Canada"*, and get the best matching jobs with a short summary and the reason each one fits.

## How it works

```
query ──► Claude Haiku: extract filters (remote/hybrid/onsite, country, seniority)
      ──► Qdrant hybrid search: semantic (bge-small) + keywords (BM25), merged with Reciprocal Rank Fusion (RRF)
      ──► Cross-encoder reranker: keep the best 5 jobs
      ──► Claude Sonnet: summary + match reason for each relevant job
      ──► JSON response (job details come straight from the data, not the LLM)
```

Repeated searches are answered instantly from a Redis cache, and every request can be traced step by step in [Langfuse](https://langfuse.com).

**Stack:** FastAPI, Qdrant, FastEmbed, Anthropic SDK, Redis, Langfuse, Docker

## Run it on your machine

You need [Docker](https://docs.docker.com/get-docker/), an [Anthropic API key](https://console.anthropic.com/), and a free [Kaggle](https://www.kaggle.com/) account to download the data.

**1. Clone the repo**

```bash
git clone https://github.com/ManashNepal/job-rag-api.git
cd job-rag-api
```

**2. Add the dataset**

Download [LinkedIn Software Engineering Jobs](https://www.kaggle.com/datasets/asaniczka/software-engineer-job-postings-linkedin) and put the CSV here:

```
data/linkedin_job_postings.csv
```

**3. Add your API key**

```bash
cp .env.example .env
```

Open `.env` and set `ANTHROPIC_API_KEY`. The Langfuse keys are optional, leave them empty to turn tracing off.

**4. Build the search index** (one time only, takes a while on a CPU)

```bash
docker compose up -d qdrant
docker compose --profile index run --rm indexer
```

**5. Start the API**

```bash
docker compose up -d
curl localhost:8000/health        # {"status":"ok","qdrant":"ok"}
```

**6. Search**

```bash
./invoke.sh "senior python engineer UK hybrid"
```

Or open **http://localhost:8000/docs** and try `POST /search` in the browser.

## Input and output

**Request:** `POST /search`

```json
{ "query": "senior python engineer UK hybrid" }
```

**Response** (example, shortened to one result):

```json
{
  "filters": { "work_type": "hybrid", "country": "United Kingdom", "seniority": "senior" },
  "filters_relaxed": false,
  "results": [
    {
      "title": "Senior Software Engineer",
      "company": "TravelPerk",
      "location": "Edinburgh, Scotland, United Kingdom",
      "country": "United Kingdom",
      "work_type": "hybrid",
      "job_level": "Mid senior",
      "seniority": "senior",
      "job_link": "https://www.linkedin.com/jobs/view/...",
      "summary": "Builds and scales backend services for a business travel platform using Python and Django...",
      "match_reason": "Senior, hybrid role in Edinburgh, UK with Python/Django as a core technology.",
      "score": 5.12
    }
  ],
  "cached": false
}
```

| Field | Meaning |
|---|---|
| `filters` | What was understood from the query |
| `filters_relaxed` | `true` if too few jobs matched, so the search ran without filters |
| `summary`, `match_reason` | Written by the LLM from the job posting |
| `score` | Reranker relevance score (higher is better) |
| `cached` | `true` if the answer came from the Redis cache |

## Dataset

[LinkedIn Software Engineering Jobs](https://www.kaggle.com/datasets/asaniczka/software-engineer-job-postings-linkedin) by asaniczka, licensed under the [ODC Attribution License](https://opendatacommons.org/licenses/by/1-0/). The data is not included in this repository.
