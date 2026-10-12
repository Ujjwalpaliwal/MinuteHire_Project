"""Local ONNX embedding model wrapper (all-MiniLM-L6-v2).

Embeddings run **locally** via onnxruntime, so the query path performs zero network
round-trips for retrieval — a major latency win for a voice loop. We use the quantized
ONNX export of all-MiniLM-L6-v2 (384-dim, ~23MB), downloaded once from the Hugging Face
Hub and cached, then mean-pooled and L2-normalised.

This avoids heavyweight frameworks (no torch/sklearn), keeping cold start fast and the
deployment image small.
"""

from __future__ import annotations

import threading

import numpy as np

from minutehire_project.config import settings

_session = None
_tokenizer = None
_input_names: set[str] = set()
_lock = threading.Lock()


def _load() -> None:
    global _session, _tokenizer, _input_names
    if _session is not None:
        return
    with _lock:
        if _session is not None:
            return
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        model_path = hf_hub_download(
            settings.embedding_repo, settings.embedding_onnx_file
        )
        tokenizer_path = hf_hub_download(
            settings.embedding_repo, settings.embedding_tokenizer_file
        )

        _session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=_session_options(ort),
        )
        _input_names = {i.name for i in _session.get_inputs()}

        _tokenizer = Tokenizer.from_file(tokenizer_path)
        _tokenizer.enable_truncation(max_length=settings.embedding_max_tokens)
        _tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")


def _session_options(ort):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


def _encode(texts: list[str]) -> np.ndarray:
    _load()
    assert _tokenizer is not None and _session is not None
    safe = [t if t.strip() else " " for t in texts]
    encodings = _tokenizer.encode_batch(safe)

    ids = np.array([e.ids for e in encodings], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

    feed = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in _input_names:
        feed["token_type_ids"] = np.zeros_like(ids)

    last_hidden = _session.run(None, feed)[0]  # (batch, seq, hidden)

    mask_f = mask[..., None].astype(np.float32)
    summed = (last_hidden * mask_f).sum(axis=1)
    counts = np.clip(mask_f.sum(axis=1), 1e-9, None)
    pooled = summed / counts
    norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
    return pooled / norms


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed documents for indexing."""
    if not texts:
        return []
    return _encode(texts).tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return _encode([text])[0].tolist()


def get_model() -> None:
    """Eagerly load (and download if needed) the embedding model."""
    _load()
