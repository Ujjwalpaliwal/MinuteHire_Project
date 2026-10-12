"""MinuteHire — RAG-based AI Voice Career Counselor."""

__version__ = "0.1.0"


def main() -> None:
    """Console entry point (``uv run minutehire-project``) — starts the API server."""
    from minutehire_project.main import main as _run

    _run()
