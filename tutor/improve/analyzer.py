"""Deterministic analysis of past session transcripts.

Turns JSONL transcripts into structured facts: which questions students ask on
which slide, how often the bot was interrupted, where it stalled, which answers
sounded uncertain, and per-session metrics. The LLM-powered step in improve.py
builds on this; everything here is unit-testable without a network.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tutor.transcripts import list_transcripts, load_transcript

UNCERTAIN_PATTERNS = re.compile(
    r"\b(i'?m not (entirely |completely |totally )?sure|i don'?t know|i'?m not certain|"
    r"i can'?t say for sure|not sure about that|i couldn'?t tell you)\b",
    re.IGNORECASE,
)
QUESTION_HINT = re.compile(r"\?|^(what|why|how|when|where|who|which|can|could|do|does|is|are|did)\b", re.IGNORECASE)


@dataclass
class QAPair:
    session_id: str
    slide: int
    mode: str
    question: str
    answer: str
    interrupted_bot: bool
    uncertain: bool


@dataclass
class SessionSummary:
    session_id: str
    user_turns: int = 0
    bot_turns: int = 0
    interruptions: int = 0
    pauses: int = 0
    safety_blocks: int = 0
    stalls: int = 0
    slides_presented: int = 0
    reached_qna: bool = False
    ended_gracefully: bool = False
    duration_seconds: float | None = None
    estimated_cost_usd: float | None = None
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0


@dataclass
class Analysis:
    sessions: list[SessionSummary] = field(default_factory=list)
    qa_pairs: list[QAPair] = field(default_factory=list)
    questions_by_slide: dict[int, list[str]] = field(default_factory=lambda: defaultdict(list))
    safety_categories: Counter = field(default_factory=Counter)

    @property
    def uncertain_pairs(self) -> list[QAPair]:
        return [p for p in self.qa_pairs if p.uncertain]

    def top_question_words(self, n: int = 10) -> list[tuple[str, int]]:
        stop = {
            "the", "a", "an", "is", "are", "what", "why", "how", "do", "does", "can", "of", "to", "in",
            "and", "it", "you", "i", "we", "about", "that", "this", "on", "for", "be", "there", "was",
        }
        counter: Counter = Counter()
        for pair in self.qa_pairs:
            for word in re.findall(r"[a-z']+", pair.question.lower()):
                if word not in stop and len(word) > 2:
                    counter[word] += 1
        return counter.most_common(n)

    def as_dict(self) -> dict:
        return {
            "sessions": len(self.sessions),
            "qa_pairs": len(self.qa_pairs),
            "uncertain_answers": len(self.uncertain_pairs),
            "questions_by_slide": {k: len(v) for k, v in sorted(self.questions_by_slide.items())},
            "safety_categories": dict(self.safety_categories),
            "interruptions": sum(s.interruptions for s in self.sessions),
            "pauses": sum(s.pauses for s in self.sessions),
            "stalls": sum(s.stalls for s in self.sessions),
            "sessions_reaching_qna": sum(1 for s in self.sessions if s.reached_qna),
            "top_question_words": self.top_question_words(),
        }


def analyze_entries(entries: list[dict], session_id: str = "") -> tuple[SessionSummary, list[QAPair]]:
    summary = SessionSummary(session_id=session_id)
    pairs: list[QAPair] = []
    pending_user: dict | None = None

    for entry in entries:
        kind = entry.get("type")
        if kind == "session_start":
            summary.session_id = entry.get("session_id", session_id) or session_id
        elif kind == "user":
            summary.user_turns += 1
            if entry.get("interrupted_bot"):
                summary.interruptions += 1
            pending_user = entry
        elif kind == "bot":
            summary.bot_turns += 1
            if pending_user is not None:
                answer = entry.get("text", "")
                pairs.append(
                    QAPair(
                        session_id=summary.session_id,
                        slide=int(pending_user.get("slide") or 0),
                        mode=str(pending_user.get("mode") or ""),
                        question=pending_user.get("text", ""),
                        answer=answer,
                        interrupted_bot=bool(pending_user.get("interrupted_bot")),
                        uncertain=bool(UNCERTAIN_PATTERNS.search(answer)),
                    )
                )
                pending_user = None
        elif kind == "event":
            name = entry.get("name")
            if name == "pause":
                summary.pauses += 1
            elif name == "safety_block":
                summary.safety_blocks += 1
            elif name == "action":
                action = entry.get("kind")
                if action == "present_slide":
                    summary.slides_presented += 1
                elif action == "enter_qna":
                    summary.reached_qna = True
                elif action == "continue_slide" and "stall" in str(entry.get("reason", "")):
                    summary.stalls += 1
            elif name in {"ending", "session_ending"}:
                summary.ended_gracefully = True
        elif kind == "session_end":
            metrics = entry.get("metrics") or {}
            summary.duration_seconds = metrics.get("duration_seconds")
            summary.estimated_cost_usd = metrics.get("estimated_cost_usd")
            for totals in (metrics.get("llm_tokens") or {}).values():
                summary.llm_prompt_tokens += int(totals.get("prompt", 0))
                summary.llm_completion_tokens += int(totals.get("completion", 0))

    return summary, pairs


def analyze_transcripts(paths: list[Path]) -> Analysis:
    analysis = Analysis()
    for path in paths:
        entries = load_transcript(path)
        summary, pairs = analyze_entries(entries, session_id=path.stem)
        analysis.sessions.append(summary)
        analysis.qa_pairs.extend(pairs)
        for pair in pairs:
            if QUESTION_HINT.search(pair.question.strip()):
                analysis.questions_by_slide[pair.slide].append(pair.question)
        for entry in entries:
            if entry.get("type") == "event" and entry.get("name") == "safety_block":
                analysis.safety_categories[str(entry.get("category"))] += 1
    return analysis


def analyze_directory(directory: Path | str) -> Analysis:
    return analyze_transcripts(list_transcripts(directory))
