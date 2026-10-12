"""In-process conversation memory.

Voice sessions are short and latency-sensitive, so we keep history in memory per session
rather than in a database. We retain only the last N turns to bound prompt size (and thus
LLM latency and cost) while preserving enough context for natural follow-ups.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from minutehire_project.config import settings


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class Session:
    id: str
    turns: list[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # Rolling summary of the student's profile extracted across the conversation.
    profile: dict = field(default_factory=dict)

    def add(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))
        self.updated_at = time.time()
        max_turns = settings.max_history_turns * 2
        if len(self.turns) > max_turns:
            self.turns = self.turns[-max_turns:]

    def history(self) -> list[dict]:
        return [{"role": t.role, "content": t.content} for t in self.turns]


class SessionStore:
    """Thread-safe, bounded LRU session store."""

    def __init__(self, max_sessions: int = 500, ttl_seconds: int = 60 * 60 * 3):
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()
        self._max_sessions = max_sessions
        self._ttl = ttl_seconds

    def get_or_create(self, session_id: str) -> Session:
        with self._lock:
            self._evict_expired()
            session = self._sessions.get(session_id)
            if session is None:
                session = Session(id=session_id)
                self._sessions[session_id] = session
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
            return session

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items() if now - s.updated_at > self._ttl
        ]
        for sid in expired:
            self._sessions.pop(sid, None)

    @property
    def count(self) -> int:
        return len(self._sessions)


store = SessionStore()
