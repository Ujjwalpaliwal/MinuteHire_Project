"""FastAPI application: RAG retrieval + streaming Gemini answers over SSE.

Latency design:
* Retrieval runs off the event loop (``asyncio.to_thread``).
* Retrieved sources are pushed to the client immediately, before the LLM starts.
* LLM tokens are streamed as Server-Sent Events so speech can begin early.
* The embedding model is warmed at startup so the first query is not slowed by a cold load.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from minutehire_project.config import settings
from minutehire_project.llm.gemini import LLMError, stream_answer
from minutehire_project.llm.prompts import build_user_prompt, opening_message
from minutehire_project.memory import store
from minutehire_project.rag import retrieve
from minutehire_project.rag.embeddings import get_model
from minutehire_project.rag.ingest import ensure_index
from minutehire_project.rag.vectorstore import collection_count

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("minutehire")

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.build_index_on_start:
        await asyncio.to_thread(ensure_index)
    if settings.warmup_on_start:
        # Preload the embedding model so the first real question is fast.
        await asyncio.to_thread(get_model)
    logger.info(
        "Ready. chunks=%d llm=%s model=%s",
        collection_count(),
        "configured" if settings.has_llm_key else "MISSING KEY",
        settings.gemini_model,
    )
    yield


app = FastAPI(title="MinuteHire AI Voice Career Counselor", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=128)


class ResetRequest(BaseModel):
    session_id: str = Field(default="default", max_length=128)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "chunks": collection_count(),
        "llm_configured": settings.has_llm_key,
        "model": settings.gemini_model,
    }


@app.get("/api/opening")
async def opening() -> dict:
    return {"message": opening_message()}


@app.post("/api/reset")
async def reset(req: ResetRequest) -> dict:
    store.reset(req.session_id)
    return {"status": "reset"}


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    session = store.get_or_create(req.session_id)
    question = req.message.strip()
    session.add("user", question)
    prior_turns = session.history()[:-1]  # everything before this question

    async def event_stream():
        try:
            chunks = await retrieve(question)
            sources = [
                {
                    "source": c.source,
                    "title": c.title,
                    "section": c.section,
                    "score": round(c.score, 3),
                }
                for c in chunks
            ]
            # Send sources first so the UI can render grounding immediately.
            yield _sse("sources", {"sources": sources})

            prompt = build_user_prompt(question, chunks)
            messages = prior_turns + [{"role": "user", "content": prompt}]

            full = []
            async for delta in stream_answer(messages):
                full.append(delta)
                yield _sse("token", {"text": delta})

            answer = "".join(full).strip()
            if answer:
                session.add("assistant", answer)
            yield _sse("done", {"text": answer})
        except LLMError as exc:
            logger.warning("LLM error: %s", exc)
            yield _sse("error", {"message": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected error in chat stream")
            yield _sse("error", {"message": f"Unexpected error: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main() -> None:
    """Console entry point: run the API with uvicorn."""
    import uvicorn

    uvicorn.run(
        "minutehire_project.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
