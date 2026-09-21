"""ContextAugmenter: refreshes the presenter state note (and retrieved knowledge)
in the LLM context right before every inference."""

from collections.abc import Callable

from loguru import logger
from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from tutor.controller import Mode, PresentationController
from tutor.knowledge.store import KnowledgeBase
from tutor.prompts import build_state_note, is_state_note


def last_user_text(context: LLMContext) -> str | None:
    messages = context.messages
    if not messages:
        return None
    last = messages[-1]
    if last.get("role") != "user":
        return None
    content = last.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return None


class ContextAugmenter(FrameProcessor):
    def __init__(
        self,
        controller: PresentationController,
        knowledge: KnowledgeBase | None = None,
        on_retrieval: Callable[[str, list[str]], None] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._controller = controller
        self._knowledge = knowledge
        self._on_retrieval = on_retrieval

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame) and not frame.speculation:
            await self._augment(frame.context)

        await self.push_frame(frame, direction)

    async def _augment(self, context: LLMContext) -> None:
        user_text = last_user_text(context)
        passages: list[str] = []

        if user_text:
            self._controller.on_user_message(user_text)
            if self._controller.mode is Mode.QNA:
                self._controller.on_user_activity_in_qna()
            if self._knowledge and self._knowledge.available:
                try:
                    passages = await self._knowledge.passages(user_text)
                except Exception as e:  # retrieval must never break the conversation
                    logger.warning(f"[knowledge] retrieval failed: {e}")
                    passages = []
                if self._on_retrieval:
                    self._on_retrieval(user_text, passages)

        note = build_state_note(self._controller, passages)
        messages = [m for m in context.messages if not is_state_note(m)]
        messages.append({"role": "system", "content": note})
        context.set_messages(messages)
