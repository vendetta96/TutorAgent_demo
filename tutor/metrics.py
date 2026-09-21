"""Session metrics aggregation and the console report printed after disconnect.

Framework-free: the Pipecat observer translates MetricsFrames into the record_*
calls below, so aggregation and formatting are deterministic and unit-testable.
"""

import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# USD per 1M tokens (LLM) / per 1M characters (TTS) / per second (STT). Estimates
# for the cost line only; adjust in config if pricing changes.
DEFAULT_PRICING = {
    "llm": {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4.1": (2.00, 8.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1-nano": (0.10, 0.40),
    },
    "tts_per_million_chars": 15.0,
    "stt_per_second": 0.0001,
}


@dataclass
class TokenTotals:
    prompt: int = 0
    completion: int = 0
    total: int = 0
    cached: int = 0
    calls: int = 0


@dataclass
class MetricsCollector:
    pricing: dict = field(default_factory=lambda: DEFAULT_PRICING)
    started_at: float = field(default_factory=time.monotonic)
    ended_at: float | None = None

    ttfb: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    processing: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    llm_tokens: dict[str, TokenTotals] = field(default_factory=dict)
    tts_characters: int = 0
    tts_calls: int = 0
    stt_audio_seconds: float = 0.0
    turn_latencies: list[float] = field(default_factory=list)
    events: Counter = field(default_factory=Counter)
    tool_calls: Counter = field(default_factory=Counter)
    user_turns: int = 0
    bot_turns: int = 0

    # ---- recording ------------------------------------------------------

    def record_ttfb(self, processor: str, seconds: float) -> None:
        self.ttfb[_short(processor)].append(seconds)

    def record_processing(self, processor: str, seconds: float) -> None:
        self.processing[_short(processor)].append(seconds)

    def record_llm_usage(
        self,
        model: str | None,
        prompt: int,
        completion: int,
        total: int | None = None,
        cached: int = 0,
    ) -> None:
        totals = self.llm_tokens.setdefault(model or "unknown", TokenTotals())
        totals.prompt += prompt
        totals.completion += completion
        totals.total += total if total is not None else prompt + completion
        totals.cached += cached or 0
        totals.calls += 1

    def record_tts_usage(self, characters: int) -> None:
        self.tts_characters += characters
        self.tts_calls += 1

    def record_stt_usage(self, audio_seconds: float) -> None:
        self.stt_audio_seconds += audio_seconds

    def record_turn_latency(self, seconds: float) -> None:
        self.turn_latencies.append(seconds)

    def record_event(self, name: str, count: int = 1) -> None:
        self.events[name] += count

    def record_tool_call(self, name: str) -> None:
        self.tool_calls[name] += 1

    def record_user_turn(self) -> None:
        self.user_turns += 1

    def record_bot_turn(self) -> None:
        self.bot_turns += 1

    def finish(self) -> None:
        if self.ended_at is None:
            self.ended_at = time.monotonic()

    # ---- derived --------------------------------------------------------

    @property
    def duration_seconds(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.monotonic()
        return max(0.0, end - self.started_at)

    def estimated_cost_usd(self) -> float:
        cost = 0.0
        for model, totals in self.llm_tokens.items():
            rates = self.pricing["llm"].get(model) or self.pricing["llm"].get(_base_model(model))
            if rates:
                in_rate, out_rate = rates
                cost += totals.prompt / 1e6 * in_rate + totals.completion / 1e6 * out_rate
        cost += self.tts_characters / 1e6 * self.pricing["tts_per_million_chars"]
        cost += self.stt_audio_seconds * self.pricing["stt_per_second"]
        return cost

    def report(self) -> dict:
        self.finish()
        return {
            "duration_seconds": round(self.duration_seconds, 1),
            "turns": {"user": self.user_turns, "bot": self.bot_turns},
            "latency": {
                "ttfb": {k: _stats(v) for k, v in self.ttfb.items()},
                "processing": {k: _stats(v) for k, v in self.processing.items()},
                "user_to_bot_response": _stats(self.turn_latencies),
            },
            "llm_tokens": {
                model: {
                    "calls": t.calls,
                    "prompt": t.prompt,
                    "completion": t.completion,
                    "total": t.total,
                    "cached": t.cached,
                }
                for model, t in self.llm_tokens.items()
            },
            "tts": {"calls": self.tts_calls, "characters": self.tts_characters},
            "stt": {"audio_seconds": round(self.stt_audio_seconds, 1)},
            "tool_calls": dict(self.tool_calls),
            "events": dict(self.events),
            "estimated_cost_usd": round(self.estimated_cost_usd(), 4),
        }


def render_report(report: dict, title: str = "SESSION METRICS REPORT") -> str:
    width = 64
    lines = ["", "=" * width, title.center(width), "=" * width]
    lines.append(f"Duration: {report['duration_seconds']:.1f}s   "
                 f"User turns: {report['turns']['user']}   Bot turns: {report['turns']['bot']}")

    lines.append("-" * width)
    lines.append("LATENCY (seconds)                 avg     min     max     p95   n")
    for label, table in (("TTFB", report["latency"]["ttfb"]), ("Processing", report["latency"]["processing"])):
        for proc, s in sorted(table.items()):
            lines.append(_row(f"{label} {proc}", s))
    resp = report["latency"]["user_to_bot_response"]
    if resp["n"]:
        lines.append(_row("User -> bot response", resp))

    lines.append("-" * width)
    lines.append("LLM TOKENS                    calls   prompt  completion    total")
    for model, t in report["llm_tokens"].items():
        lines.append(f"  {model:<26}{t['calls']:>7}{t['prompt']:>9}{t['completion']:>12}{t['total']:>9}")
    if not report["llm_tokens"]:
        lines.append("  (no LLM usage recorded)")

    lines.append("-" * width)
    lines.append(f"TTS: {report['tts']['calls']} requests, {report['tts']['characters']} characters   "
                 f"STT: {report['stt']['audio_seconds']:.1f}s of audio")

    if report["tool_calls"]:
        lines.append("-" * width)
        lines.append("TOOL CALLS: " + ", ".join(f"{k}={v}" for k, v in sorted(report["tool_calls"].items())))
    if report["events"]:
        lines.append("-" * width)
        lines.append("EVENTS: " + ", ".join(f"{k}={v}" for k, v in sorted(report["events"].items())))

    lines.append("-" * width)
    lines.append(f"Estimated cost: ${report['estimated_cost_usd']:.4f} USD (list prices; approximate)")
    lines.append("=" * width)
    return "\n".join(lines)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0, "n": 0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "avg": round(statistics.fmean(values), 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
        "p95": round(ordered[p95_index], 3),
        "n": len(values),
    }


def _row(label: str, s: dict) -> str:
    return f"  {label:<30}{s['avg']:>7.3f} {s['min']:>7.3f} {s['max']:>7.3f} {s['p95']:>7.3f} {s['n']:>3}"


def _short(processor: str) -> str:
    # Pipecat names processors like "OpenAILLMService#0"; keep the service name only.
    return processor.split("#")[0].replace("Service", "")


def _base_model(model: str) -> str:
    # Map dated snapshots like "gpt-4o-2024-08-06" to their base name.
    parts = model.split("-")
    while parts and parts[-1].isdigit():
        parts.pop()
    return "-".join(parts)
