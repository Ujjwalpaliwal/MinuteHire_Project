FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KB_DIR=/app/data/knowledge_base \
    CHROMA_DIR=/app/data/chroma \
    HF_HOME=/app/.cache/huggingface \
    ANONYMIZED_TELEMETRY=false

WORKDIR /app

# onnxruntime needs GNU OpenMP; slim images do not ship it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the package and its dependencies.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

# Knowledge base is baked in; the vector index and embedding model are built at
# image build time so cold starts do no network or CPU work. This RUN needs
# network access to download the ONNX model from the Hugging Face Hub.
COPY data/knowledge_base ./data/knowledge_base
RUN python -c "from minutehire_project.rag.ingest import build_index; build_index()" \
    && find /app/.cache -name "*.lock" -delete

# Model + index are in the image now: forbid runtime hub access entirely.
ENV HF_HUB_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "minutehire_project.main:app", "--host", "0.0.0.0", "--port", "8000"]
