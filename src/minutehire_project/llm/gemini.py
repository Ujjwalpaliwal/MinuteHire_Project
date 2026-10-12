"""Streaming Gemini Flash client over the REST API.

We call the REST streaming endpoint directly with httpx rather than pulling in a heavy SDK.
Streaming means the first tokens reach the browser (and can start being spoken) in a few
hundred milliseconds instead of waiting for the whole answer.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

import httpx

from minutehire_project.config import settings
from minutehire_project.llm.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM cannot be reached or returns an error."""


def _build_payload(messages: list[dict], temperature: float, max_tokens: int) -> dict:
    contents = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    return {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": 0.95,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ],
    }


def _extract_text(chunk: dict) -> str:
    candidates = chunk.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts)


async def stream_answer(
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AsyncIterator[str]:
    """Yield incremental text deltas from Gemini.

    `messages` is a list of {"role": "user"|"assistant", "content": str} in order.
    """
    if not settings.has_llm_key:
        raise LLMError(
            "GEMINI_API_KEY is not set. Add it to your .env file to enable responses."
        )

    url = (
        f"{settings.gemini_base_url}/models/{settings.gemini_model}"
        f":streamGenerateContent?alt=sse"
    )
    payload = _build_payload(
        messages,
        temperature if temperature is not None else settings.temperature,
        max_tokens if max_tokens is not None else settings.max_output_tokens,
    )

    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                async with client.stream(
                    "POST",
                    url,
                    headers={
                        "x-goog-api-key": settings.gemini_api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        detail = body.decode("utf-8", "ignore")[:400]
                        # Retry transient errors (rate limit / server errors).
                        if response.status_code in {429, 500, 502, 503, 504} and attempt < attempts:
                            await asyncio.sleep(1.5 * attempt)
                            continue
                        raise LLMError(f"Gemini API error {response.status_code}: {detail}")

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        text = _extract_text(chunk)
                        if text:
                            yield text
                    return
        except httpx.HTTPError as exc:
            if attempt < attempts:
                await asyncio.sleep(1.5 * attempt)
                continue
            raise LLMError(f"Network error calling Gemini: {exc}") from exc
