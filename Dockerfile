# Stage 1: install dependencies 
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.10.6 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project


# Stage 2: runtime 
FROM python:3.12-slim

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/models/fastembed \
    HF_HOME=/app/models/huggingface

RUN useradd --create-home --uid 1000 app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY run_indexing.py ./

# Download the embedding, BM25 and reranker models at build time, 
# so containers start without downloading anything.
RUN python -c "\
from src.indexing.embedder import get_dense_model, get_sparse_model, get_reranker_model; \
from src.indexing.chunking import get_tokenizer; \
get_dense_model(); get_sparse_model(); get_reranker_model(); get_tokenizer()" \
    && chown -R app:app /app/models

# Models are in the image now, so never reach out to Hugging Face at runtime
ENV HF_HUB_OFFLINE=1

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
