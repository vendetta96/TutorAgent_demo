"""Session transcript recording (JSONL) and loading.

Every session is written to data/transcripts/<session_id>.jsonl as a stream of
events. These files feed the offline improvement loop in tutor.improve.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tutor.safety import SafetyGuard

DEFAULT_TRANSCRIPT_DIR = Path("data/transcripts")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class TranscriptRecorder:
    def __init__(self, directory: Path | str = DEFAULT_TRANSCRIPT_DIR, session_id: str | None = None):
        self.directory = Path(directory)
        self.session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.path = self.directory / f"{self.session_id}.jsonl"
        self._entries: list[dict] = []
        self._opened = False

    def _write(self, entry: dict) -> None:
        entry = {"ts": _now(), **entry}
        self._entries.append(entry)
        if not self._opened:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._opened = True
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def session_started(self, **meta) -> None:
        self._write({"type": "session_start", "session_id": self.session_id, **meta})

    def user(self, text: str, *, slide: int, mode: str, interrupted: bool = False) -> None:
        self._write(
            {
                "type": "user",
                "text": SafetyGuard.redact(text),
                "slide": slide,
                "mode": mode,
                "interrupted_bot": interrupted,
            }
        )

    def bot(self, text: str, *, slide: int, mode: str, interrupted: bool = False) -> None:
        self._write({"type": "bot", "text": text, "slide": slide, "mode": mode, "interrupted": interrupted})

    def event(self, name: str, **data) -> None:
        self._write({"type": "event", "name": name, **data})

    def session_ended(self, metrics: dict | None = None, controller_stats: dict | None = None) -> None:
        self._write({"type": "session_end", "metrics": metrics or {}, "controller": controller_stats or {}})

    @property
    def entries(self) -> list[dict]:
        return list(self._entries)


def load_transcript(path: Path | str) -> list[dict]:
    entries = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def list_transcripts(directory: Path | str = DEFAULT_TRANSCRIPT_DIR) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob("*.jsonl") if p.is_file())
