"""Build the vector index from the knowledge base.

Usage:
    uv run python scripts/build_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minutehire_project.rag.ingest import build_index  # noqa: E402


if __name__ == "__main__":
    total = build_index(rebuild=True)
    print(f"Indexed {total} chunks.")
