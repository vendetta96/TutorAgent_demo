"""SafetyProcessor: screens student transcriptions before they reach the LLM.

Blocked messages are dropped, a fixed safe reply is spoken via TTSSpeakFrame,
and the LLM context receives a short system note so the conversation stays
coherent without ever seeing the blocked content.
"""

from collections.abc import Awaitable, Callable

from loguru import logger
from pipecat.frames.frames import Frame, LLMMessagesAppendFrame, TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy

from tutor.safety import SafetyGuard, SafetyVerdict


class SafetyProcessor(FrameProcessor):
    def __init__(
        self,
        guard: SafetyGuard | None = None,
        on_block: Callable[[str, SafetyVerdict], None] | None = None,
        moderator: Callable[[str], Awaitable[SafetyVerdict]] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._guard = guard or SafetyGuard()
        self._on_block = on_block
        self._moderator = moderator
        self.blocked_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            verdict = self._guard.check(frame.text)
            if verdict.allowed and self._moderator is not None:
                try:
                    verdict = await self._moderator(frame.text)
                except Exception as e:  # moderation must never break the lesson
                    logger.warning(f"[safety] moderation call failed: {e}")
                    verdict = SafetyVerdict(allowed=True)
            if not verdict.allowed:
                await self._block(frame.text, verdict)
                return

        await self.push_frame(frame, direction)

    async def _block(self, text: str, verdict: SafetyVerdict) -> None:
        self.blocked_count += 1
        category = verdict.category.value if verdict.category else "unknown"
        logger.warning(f"[safety] blocked student message (category={category}, matched={verdict.matched!r})")
        if self._on_block:
            self._on_block(text, verdict)

        note = {
            "role": "system",
            "content": (
                f"[SAFETY] A student message was blocked by the safety filter (category: {category}). "
                "You replied with a fixed safe response. Do not refer to the blocked content; "
                "continue the lesson normally."
            ),
        }
        await self.push_frame(LLMMessagesAppendFrame(messages=[note], run_llm=False))
        assert verdict.reply is not None
        await self.push_frame(TTSSpeakFrame(text=verdict.reply))


class PausedUserMuteStrategy(BaseUserMuteStrategy):
    """Mutes the student (VAD, interruptions, transcriptions) while the lesson is paused."""

    def __init__(self, is_paused: Callable[[], bool]):
        super().__init__()
        self._is_paused = is_paused

    async def process_frame(self, frame: Frame) -> bool:
        await super().process_frame(frame)
        return self._is_paused()
