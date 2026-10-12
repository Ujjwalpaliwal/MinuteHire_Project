FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KB_DIR=/app/data/knowledge_base \
    CHROMA_DIR=/app/data/chroma \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install the package and its dependencies.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

# Knowledge base is needed at runtime to build the vector index.
COPY data/knowledge_base ./data/knowledge_base

EXPOSE 8000

CMD ["uvicorn", "minutehire_project.main:app", "--host", "0.0.0.0", "--port", "8000"]
