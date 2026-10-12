# MinuteHire — RAG-based AI Voice Career Counselor

A real-time **voice** career counselor for students. You speak, it transcribes, retrieves
grounded career guidance from a curated knowledge base, generates a personalized answer with
Gemini Flash (streamed), and speaks the response back.

The goal of this project is a **low-latency, RAG-grounded conversational agent** — not a
fancy UI.

---

## Architecture

```
🎤 User speaks
   ↓
Browser Speech Recognition (Web Speech API)      ← STT, no network round-trip
   ↓
POST /api/chat  (FastAPI)
   ↓
Local ONNX embeddings  →  ChromaDB vector search   ← retrieval, local & fast
   ↓
Gemini Flash  (streamed token-by-token over SSE)   ← grounded generation
   ↓
Browser Speech Synthesis (streaming, sentence-by-sentence)   ← TTS starts early
   ↓
🔊 AI speaks
```

| Layer | Choice | Why |
|---|---|---|
| Speech-to-text | Browser Web Speech API | Zero backend audio infra, no per-request cost, lowest latency |
| Text-to-speech | Browser Speech Synthesis API | Same: local, instant start |
| Embeddings | ONNX `all-MiniLM-L6-v2` (384-d, quantized) | Fully local, no network round-trip, tiny (~23 MB), fast on CPU |
| Vector store | ChromaDB (persistent) | Embedded DB, no server, cosine search; swappable for Pinecone/FAISS |
| LLM | Gemini Flash (REST, streaming) | Fast + cheap, native token streaming, strong instruction following |
| Backend | FastAPI + SSE | Async, streaming responses, minimal surface area |

The vector store and LLM are isolated behind small modules (`rag/vectorstore.py`,
`llm/gemini.py`), so replacing Chroma with Pinecone or Gemini with another provider is a
local change.

---

## RAG pipeline

1. **Knowledge base** (`data/knowledge_base/*.md`) — 12 curated documents across career
   paths, skills, education, progression, job roles, industry, counseling frameworks,
   job search and learning paths.
2. **Chunking** (`rag/chunking.py`) — markdown is split **on headings first** to keep chunks
   topically coherent, then paragraphs are packed up to ~220 words with a 40-word overlap so
   context spanning a boundary is preserved. Each chunk is prefixed with its heading so it is
   self-describing.
3. **Embeddings** (`rag/embeddings.py`) — local ONNX MiniLM, mean-pooled + L2-normalised.
4. **Vector store** (`rag/vectorstore.py`) — ChromaDB with cosine space.
5. **Retrieval** (`rag/retriever.py`) — over-fetch candidates, deduplicate near-identical
   chunks, return top-k with a similarity score.
6. **Grounding** (`llm/prompts.py`) — retrieved chunks are injected into the prompt with an
   explicit instruction to ground factual claims in the context and to admit when context is
   missing.

Build the index:

```bash
python scripts/build_index.py      # or: uv run python scripts/build_index.py
```

The server also builds the index automatically on first start if the collection is empty.

---

## Latency design

Voice latency is dominated by *time-to-first-audio*, so every stage is optimized for it:

- **Local STT/TTS** — the browser handles both, so there is no audio upload/download and no
  third-party speech API latency or cost.
- **Local embeddings** — the query embedding is computed on-device (no network hop before
  retrieval).
- **Async, non-blocking retrieval** — embedding + Chroma search run in a worker thread via
  `asyncio.to_thread`, so the event loop keeps serving other sessions.
- **Streamed generation (SSE)** — tokens are pushed the moment Gemini produces them; the UI
  renders text progressively instead of waiting for the full answer.
- **Speech starts before the answer finishes** — the frontend flushes complete sentences to
  the speech synthesizer as tokens arrive, so audio begins after the first sentence rather
  than after the whole response.
- **Sources first** — retrieved sources are emitted as the first SSE event, before the LLM
  call, so grounding is visible immediately.
- **Bounded context** — only the last N conversation turns are kept, limiting prompt size
  (and therefore LLM latency).
- **Warm start** — the embedding model is loaded at startup so the first real question is not
  slowed by a cold model load.
- **Small, quantized model** — fast CPU inference and fast cold start.

---

## Running locally

Requirements: Python 3.13 and [`uv`](https://docs.astral.sh/uv/) (or plain `pip`).

```bash
# 1. Install dependencies
uv sync

# 2. Configure your key
copy .env.example .env        # then edit .env and set GEMINI_API_KEY

# 3. Build the index (optional — server builds it on first start)
uv run python scripts/build_index.py

# 4. Run
uv run uvicorn minutehire_project.main:app --reload --port 8000
```

Open <http://localhost:8000>, press the mic, and talk. Chrome or Edge is recommended for
voice input; on unsupported browsers (e.g. Firefox) a text input fallback appears.

> Get a free Gemini API key at <https://aistudio.google.com/apikey>.

---

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Status, indexed chunk count, whether the LLM key is configured |
| `GET` | `/api/opening` | The counselor's greeting |
| `POST` | `/api/chat` | SSE stream: `sources`, `token`, `done`, `error` events |
| `POST` | `/api/reset` | Clear a session's conversation memory |
| `GET` | `/` | Voice UI |

`POST /api/chat` body: `{ "message": "...", "session_id": "..." }`

---

## Deployment

### Render (recommended)

1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, point at the repo (uses `render.yaml`), or create a
   **Web Service** with runtime **Docker**.
3. Set the `GEMINI_API_KEY` environment variable.
4. Deploy. The app builds the index on first boot and is ready at the live URL.

### Docker (any host)

```bash
docker build -t minutehire .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key minutehire
```

The image is CPU-only and lightweight (no torch): `onnxruntime` + ChromaDB + FastAPI.

---

## Project layout

```
data/knowledge_base/            Curated career knowledge base (12 markdown docs)
scripts/build_index.py          CLI to (re)build the vector index
src/minutehire_project/
  config.py                     Env-driven settings
  main.py                       FastAPI app, SSE chat endpoint, static serving
  memory.py                     Bounded, per-session conversation memory
  llm/
    prompts.py                  Counselor persona + grounding prompt
    gemini.py                   Streaming Gemini REST client
  rag/
    chunking.py                 Heading-aware chunking with overlap
    embeddings.py               Local ONNX MiniLM embeddings
    vectorstore.py              ChromaDB wrapper
    ingest.py                   Load → chunk → embed → index
    retriever.py                Async retrieval + dedup
  static/index.html             Voice UI (Web Speech API STT/TTS)
```

---

## Design notes for reviewers

- **Model selection:** Flash-class model for latency/cost; a small local embedding model is
  sufficient because the knowledge base is domain-specific and short-form.
- **Chunking:** heading-first, overlap-based. Keeps each retrieved unit coherent, which
  improves precision and reduces hallucination.
- **Grounding:** the system prompt forbids inventing salaries/statistics and requires the
  model to defer to retrieved context — the core anti-hallucination control for a counselor.
- **Failure handling:** missing API key, rate limits (retried with backoff) and network errors
  surface as clean `error` SSE events instead of broken streams.
- **Swappability:** local embeddings and an embedded vector store keep the whole retrieval
  path off the network; Pinecone/Gemini can be substituted without touching the API layer.
