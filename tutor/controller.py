"""Pure presentation state machine. No framework dependencies, fully deterministic.

The controller never touches timers or the pipeline. It receives events and
returns `Action`s describing what the driver should do next.
"""

from dataclasses import dataclass, field
from enum import Enum

from tutor.slides import DECK, Slide


class Mode(str, Enum):
    IDLE = "idle"
    PRESENTING = "presenting"
    QNA = "qna"
    ENDED = "ended"


class ActionKind(str, Enum):
    PRESENT_SLIDE = "present_slide"
    CONTINUE_SLIDE = "continue_slide"
    ENTER_QNA = "enter_qna"
    IDLE_PROMPT = "idle_prompt"
    END_SESSION = "end_session"
    SCHEDULE_ADVANCE = "schedule_advance"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    slide: Slide | None = None
    reason: str = ""


@dataclass
class ControllerStats:
    slides_presented: int = 0
    user_turns: int = 0
    interruptions: int = 0
    jumps: int = 0
    pauses: int = 0
    idle_prompts: int = 0
    stalls_recovered: int = 0


@dataclass
class PresentationController:
    deck: tuple[Slide, ...] = DECK
    max_idle_prompts: int = 2

    mode: Mode = Mode.IDLE
    slide_number: int = 0
    paused: bool = False
    bot_speaking: bool = False
    user_speaking: bool = False
    interrupted_mid_slide: bool = False
    slide_started_speaking: bool = False
    # Set when a fixed safety reply was spoken instead of an LLM answer, so the
    # interrupted slide still gets finished before moving on.
    resume_after_reply: bool = False
    completed_slides: set[int] = field(default_factory=set)
    stats: ControllerStats = field(default_factory=ControllerStats)
    history: list[str] = field(default_factory=list)

    # ---- properties -----------------------------------------------------

    @property
    def current_slide(self) -> Slide | None:
        if 1 <= self.slide_number <= len(self.deck):
            return self.deck[self.slide_number - 1]
        return None

    @property
    def total_slides(self) -> int:
        return len(self.deck)

    @property
    def is_last_slide(self) -> bool:
        return self.slide_number >= len(self.deck)

    # ---- lifecycle ------------------------------------------------------

    def start(self) -> Action:
        self._log("start")
        self.mode = Mode.PRESENTING
        return self._present(1, reason="start")

    def end(self) -> None:
        self._log("end")
        self.mode = Mode.ENDED

    # ---- speech events --------------------------------------------------

    def on_bot_started_speaking(self) -> None:
        self.bot_speaking = True
        if self.mode is Mode.PRESENTING:
            self.slide_started_speaking = True

    def on_bot_stopped_speaking(self) -> Action | None:
        self.bot_speaking = False
        if self.mode is Mode.PRESENTING and not self.paused and not self.user_speaking:
            return Action(ActionKind.SCHEDULE_ADVANCE, self.current_slide, "bot finished speaking")
        return None

    def on_user_started_speaking(self) -> None:
        self.user_speaking = True
        if self.mode is Mode.PRESENTING and self.bot_speaking:
            self.interrupted_mid_slide = True
            self.stats.interruptions += 1
            self._log(f"interrupted on slide {self.slide_number}")

    def on_user_stopped_speaking(self) -> None:
        self.user_speaking = False

    def on_user_message(self, text: str) -> None:
        if text.strip():
            self.stats.user_turns += 1

    # ---- slide navigation -----------------------------------------------

    def advance(self) -> Action | None:
        """Called when the silence gap after a slide elapsed. Deterministic progression."""
        if self.mode is not Mode.PRESENTING or self.paused:
            return None
        if self.slide_number == 0:
            return self._present(1, reason="advance from idle")
        if self.resume_after_reply:
            self.resume_after_reply = False
            if self.interrupted_mid_slide:
                self.interrupted_mid_slide = False
                self._log(f"continue slide {self.slide_number} after safety reply")
                return Action(ActionKind.CONTINUE_SLIDE, self.current_slide, "finish slide after safety reply")
        self.completed_slides.add(self.slide_number)
        if self.is_last_slide:
            return self._enter_qna(reason="last slide finished")
        return self._present(self.slide_number + 1, reason="auto-advance")

    def next_slide(self) -> Action:
        """Explicit user request to move forward (works in any mode)."""
        if self.slide_number == 0:
            return self._present(1, reason="user: next slide")
        self.completed_slides.add(self.slide_number)
        self.stats.jumps += 1
        if self.is_last_slide:
            return self._enter_qna(reason="user: next after last slide")
        self.mode = Mode.PRESENTING
        return self._present(self.slide_number + 1, reason="user: next slide")

    def go_to_slide(self, number: int) -> Action:
        if number < 1 or number > len(self.deck):
            raise ValueError(f"Slide {number} does not exist; the deck has {len(self.deck)} slides")
        self.stats.jumps += 1
        self.mode = Mode.PRESENTING
        return self._present(number, reason=f"user: go to slide {number}")

    def find_slide_by_topic(self, query: str) -> Slide | None:
        q = query.lower()
        best: tuple[int, Slide] | None = None
        for slide in self.deck:
            score = sum(1 for k in slide.keywords if k in q)
            if slide.title.lower() in q:
                score += 3
            if score and (best is None or score > best[0]):
                best = (score, slide)
        return best[1] if best else None

    # ---- pause / resume -------------------------------------------------

    def pause(self) -> bool:
        if self.paused or self.mode is Mode.ENDED:
            return False
        self.paused = True
        self.stats.pauses += 1
        self._log("pause")
        return True

    def resume(self) -> Action | None:
        if not self.paused:
            return None
        self.paused = False
        self._log("resume")
        if self.mode is Mode.PRESENTING and not self.bot_speaking and self.slide_started_speaking:
            return Action(ActionKind.SCHEDULE_ADVANCE, self.current_slide, "resumed after slide finished")
        return None

    # ---- recovery -------------------------------------------------------

    def on_stall(self) -> Action | None:
        """Nothing has happened for too long while presenting: nudge the LLM to continue."""
        if self.mode is not Mode.PRESENTING or self.paused or self.bot_speaking or self.user_speaking:
            return None
        self.stats.stalls_recovered += 1
        self._log(f"stall on slide {self.slide_number}")
        if not self.slide_started_speaking:
            return self._present(self.slide_number or 1, reason="stall: slide never started")
        return Action(ActionKind.CONTINUE_SLIDE, self.current_slide, "stall: continue slide")

    def on_qna_idle(self) -> Action | None:
        if self.mode is not Mode.QNA or self.paused:
            return None
        self.stats.idle_prompts += 1
        if self.stats.idle_prompts > self.max_idle_prompts:
            self._log("idle limit reached")
            return Action(ActionKind.END_SESSION, reason="no questions after repeated prompts")
        return Action(ActionKind.IDLE_PROMPT, reason=f"idle prompt {self.stats.idle_prompts}")

    def on_user_activity_in_qna(self) -> None:
        self.stats.idle_prompts = 0

    # ---- snapshot -------------------------------------------------------

    def snapshot(self) -> dict:
        slide = self.current_slide
        return {
            "mode": self.mode.value,
            "paused": self.paused,
            "slide_number": self.slide_number,
            "total_slides": self.total_slides,
            "slide_title": slide.title if slide else None,
            "talking_points": list(slide.talking_points) if slide else [],
            "completed_slides": sorted(self.completed_slides),
            "interrupted_mid_slide": self.interrupted_mid_slide,
            "stats": self.stats.__dict__.copy(),
        }

    # ---- internals ------------------------------------------------------

    def _present(self, number: int, *, reason: str) -> Action:
        self.slide_number = number
        self.interrupted_mid_slide = False
        self.resume_after_reply = False
        self.slide_started_speaking = False
        self.stats.slides_presented += 1
        self._log(f"present slide {number} ({reason})")
        return Action(ActionKind.PRESENT_SLIDE, self.current_slide, reason)

    def _enter_qna(self, *, reason: str) -> Action:
        self.mode = Mode.QNA
        self.interrupted_mid_slide = False
        self.stats.idle_prompts = 0
        self._log(f"enter qna ({reason})")
        return Action(ActionKind.ENTER_QNA, self.current_slide, reason)

    def _log(self, entry: str) -> None:
        self.history.append(entry)
