"""PresentationDriver: the Pipecat-side executor for the PresentationController.

It observes pipeline frames, feeds events to the pure controller, and carries out
the returned Actions (inject slide prompts, schedule the slide gap, pause/resume,
end the session) plus the timers the controller deliberately does not own.
"""

import asyncio
import time
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.task import PipelineTask

from tutor.controller import Action, ActionKind, Mode, PresentationController
from tutor.metrics import MetricsCollector
from tutor.processors.metrics_observer import FrameDeduper
from tutor.processors.pause_gate import PauseGate
from tutor.prompts import (
    continue_slide_message,
    enter_qna_message,
    goodbye_message,
    idle_prompt_message,
    slide_message,
)
from tutor.transcripts import TranscriptRecorder


class PresentationDriver(BaseObserver):
    def __init__(
        self,
        controller: PresentationController,
        pause_gate: PauseGate,
        *,
        recorder: TranscriptRecorder | None = None,
        collector: MetricsCollector | None = None,
        slide_gap_secs: float = 1.2,
        stall_secs: float = 8.0,
        qna_idle_secs: float = 25.0,
        end_grace_secs: float = 20.0,
        start_fallback_secs: float = 3.0,
        watchdog_tick_secs: float = 1.0,
    ):
        super().__init__()
        self.controller = controller
        self.gate = pause_gate
        self.recorder = recorder
        self.collector = collector
        self.slide_gap_secs = slide_gap_secs
        self.stall_secs = stall_secs
        self.qna_idle_secs = qna_idle_secs
        self.end_grace_secs = end_grace_secs
        self.start_fallback_secs = start_fallback_secs
        self.watchdog_tick_secs = watchdog_tick_secs

        self._task: PipelineTask | None = None
        self._dedupe = FrameDeduper()
        self._advance_timer: asyncio.TimerHandle | None = None
        self._watchdog: asyncio.Task | None = None
        self._end_timer: asyncio.TimerHandle | None = None
        self._llm_busy = 0
        self._tool_calls_pending = 0
        self._last_activity = time.monotonic()
        self._ending = False
        self._started = False
        self._state_listeners: list[Any] = []

    # ---- wiring ---------------------------------------------------------

    def set_task(self, task: PipelineTask) -> None:
        self._task = task

    def add_state_listener(self, listener) -> None:
        """listener(snapshot: dict) -> Awaitable; called whenever the state changes."""
        self._state_listeners.append(listener)

    @property
    def busy(self) -> bool:
        return self._llm_busy > 0 or self._tool_calls_pending > 0

    # ---- frame observation ---------------------------------------------

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if not self._dedupe.first_time(frame):
            return

        if isinstance(frame, StartFrame):
            self._start_watchdog()
            # Normally the lesson starts on the client-ready handshake; this is the fallback.
            loop = asyncio.get_running_loop()
            loop.call_later(self.start_fallback_secs, lambda: asyncio.create_task(self.start()))
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._touch()
            self._cancel_advance()
            self.controller.on_bot_started_speaking()
            await self._broadcast_state()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._touch()
            action = self.controller.on_bot_stopped_speaking()
            if self._ending:
                if not self.busy:
                    await self._finish_session()
            elif action:
                await self.execute(action)
            await self._broadcast_state()
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._touch()
            self._cancel_advance()
            self.controller.on_user_started_speaking()
            await self._broadcast_state()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._touch()
            self.controller.on_user_stopped_speaking()
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._touch()
            self._llm_busy += 1
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._touch()
            self._llm_busy = max(0, self._llm_busy - 1)
        elif isinstance(frame, FunctionCallsStartedFrame):
            self._touch()
            self._cancel_advance()
            self._tool_calls_pending += len(frame.function_calls)
        elif isinstance(frame, FunctionCallResultFrame):
            self._touch()
            self._tool_calls_pending = max(0, self._tool_calls_pending - 1)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            self.shutdown()

    # ---- actions --------------------------------------------------------

    async def start(self) -> None:
        if self._started or self._task is None:
            return
        self._started = True
        self._start_watchdog()
        logger.info("[driver] starting the lesson")
        await self.execute(self.controller.start())

    async def execute(self, action: Action | None) -> None:
        if action is None or self._task is None:
            return
        self._touch()
        logger.info(f"[driver] {action.kind.value}: {action.reason}")
        if self.recorder:
            self.recorder.event(
                "action",
                kind=action.kind.value,
                reason=action.reason,
                slide=action.slide.number if action.slide else None,
            )

        if action.kind is ActionKind.SCHEDULE_ADVANCE:
            self._schedule_advance()
            return

        if action.kind is ActionKind.PRESENT_SLIDE and action.slide:
            first = action.reason == "start"
            await self._inject(slide_message(action.slide, first=first))
        elif action.kind is ActionKind.CONTINUE_SLIDE and action.slide:
            await self._inject(continue_slide_message(action.slide))
        elif action.kind is ActionKind.ENTER_QNA:
            await self._inject(enter_qna_message())
        elif action.kind is ActionKind.IDLE_PROMPT:
            await self._inject(idle_prompt_message(self.controller.stats.idle_prompts))
        elif action.kind is ActionKind.END_SESSION:
            await self.begin_ending(inject_goodbye=True)

        await self._broadcast_state()

    async def _inject(self, content: str) -> None:
        assert self._task is not None
        await self._task.queue_frames(
            [LLMMessagesAppendFrame(messages=[{"role": "system", "content": content}], run_llm=True)]
        )

    async def begin_ending(self, *, inject_goodbye: bool) -> None:
        """Wind down: say goodbye (unless the LLM is already doing so), then end after the bot stops."""
        if self._ending:
            return
        self._ending = True
        self.controller.end()
        self._cancel_advance()
        if self.recorder:
            self.recorder.event("ending", inject_goodbye=inject_goodbye)
        if inject_goodbye:
            await self._inject(goodbye_message())
        loop = asyncio.get_running_loop()
        self._end_timer = loop.call_later(
            self.end_grace_secs, lambda: asyncio.create_task(self._finish_session())
        )

    async def _finish_session(self) -> None:
        if self._task is None:
            return
        if self._end_timer:
            self._end_timer.cancel()
            self._end_timer = None
        logger.info("[driver] ending session")
        if self.recorder:
            self.recorder.event("session_ending")
        await self._task.stop_when_done()

    # ---- pause / resume -------------------------------------------------

    async def pause(self) -> bool:
        if not self.controller.pause():
            return False
        self._cancel_advance()
        await self.gate.pause()
        if self.collector:
            self.collector.record_event("pause")
        if self.recorder:
            self.recorder.event("pause", slide=self.controller.slide_number)
        await self._broadcast_state()
        return True

    async def resume(self) -> bool:
        if not self.controller.paused:
            return False
        action = self.controller.resume()
        await self.gate.resume()
        self._touch()
        if self.recorder:
            self.recorder.event("resume", slide=self.controller.slide_number)
        await self.execute(action)
        await self._broadcast_state()
        return True

    # ---- timers ---------------------------------------------------------

    def _schedule_advance(self) -> None:
        self._cancel_advance()
        loop = asyncio.get_running_loop()
        self._advance_timer = loop.call_later(
            self.slide_gap_secs, lambda: asyncio.create_task(self._on_advance_timer())
        )

    def _cancel_advance(self) -> None:
        if self._advance_timer:
            self._advance_timer.cancel()
            self._advance_timer = None

    async def _on_advance_timer(self) -> None:
        self._advance_timer = None
        if self.busy or self.controller.bot_speaking or self.controller.user_speaking:
            # Something is still in flight (e.g. a tool call re-running the LLM); check again shortly.
            self._schedule_advance()
            return
        await self.execute(self.controller.advance())

    def _start_watchdog(self) -> None:
        if self._watchdog is None:
            self._watchdog = asyncio.create_task(self._watchdog_loop())

    async def _watchdog_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.watchdog_tick_secs)
                await self._check_idle()
        except asyncio.CancelledError:
            pass

    async def _check_idle(self) -> None:
        c = self.controller
        if c.paused or c.bot_speaking or c.user_speaking or self.busy or self._ending:
            return
        if self._advance_timer is not None:
            return
        idle = time.monotonic() - self._last_activity
        if c.mode is Mode.PRESENTING and idle >= self.stall_secs:
            action = c.on_stall()
            if action:
                if self.collector:
                    self.collector.record_event("stall_recovered")
                await self.execute(action)
        elif c.mode is Mode.QNA and idle >= self.qna_idle_secs:
            action = c.on_qna_idle()
            if action:
                await self.execute(action)

    def _touch(self) -> None:
        self._last_activity = time.monotonic()

    def shutdown(self) -> None:
        self._cancel_advance()
        if self._end_timer:
            self._end_timer.cancel()
            self._end_timer = None
        if self._watchdog:
            self._watchdog.cancel()
            self._watchdog = None

    # ---- UI state -------------------------------------------------------

    async def _broadcast_state(self) -> None:
        snapshot = self.controller.snapshot()
        snapshot["bot_speaking"] = self.controller.bot_speaking
        for listener in self._state_listeners:
            try:
                await listener(snapshot)
            except Exception as e:
                logger.warning(f"[driver] state listener failed: {e}")
