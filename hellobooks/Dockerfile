# ────────────────────────────────────────────────────────────────────────────
# Hellobooks AI — Dockerfile
# Multi-stage build: keeps final image lean
# ────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ──────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ───────────────────────────────────────────────────────
COPY knowledge_base/ ./knowledge_base/
COPY src/            ./src/

# Pre-download the embedding model at build time so the container works offline
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('all-MiniLM-L6-v2')"

# ── Pre-build vector store (optional; speeds up first run) ───────────────────
# Uncomment the next line to embed docs at image build time
# RUN python src/rag.py --rebuild --question "test" || true

# ── Runtime configuration ────────────────────────────────────────────────────
# Pass your API key at runtime:
#   docker run -e OPENAI_API_KEY=sk-... hellobooks-ai
ENV LLM_PROVIDER=openai
ENV LLM_MODEL=gpt-3.5-turbo
ENV PYTHONUNBUFFERED=1

# Default: interactive CLI
ENTRYPOINT ["python", "src/rag.py"]
# To ask a single question non-interactively:
#   docker run hellobooks-ai --question "What is a balance sheet?"
