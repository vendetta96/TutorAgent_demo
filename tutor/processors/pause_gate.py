"""PauseGate: holds every downstream bot-output frame while paused, releases them
in order on resume so the agent continues exactly where it stopped."""

from loguru import logger
from pipecat.frames.frames import EndFrame, Frame, InterruptionFrame, SystemFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class PauseGate(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._paused = False
        self._buffer: list[Frame] = []

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def buffered(self) -> int:
        return len(self._buffer)

    async def pause(self) -> None:
        if not self._paused:
            self._paused = True
            logger.info("[pause-gate] paused")

    async def resume(self) -> None:
        if not self._paused:
            return
        self._paused = False
        pending, self._buffer = self._buffer, []
        logger.info(f"[pause-gate] resumed, releasing {len(pending)} buffered frames")
        for frame in pending:
            await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    def clear(self) -> int:
        dropped = len(self._buffer)
        self._buffer = []
        return dropped

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.UPSTREAM or isinstance(frame, SystemFrame):
            if isinstance(frame, InterruptionFrame) and self._buffer:
                logger.info(f"[pause-gate] interruption, dropping {self.clear()} buffered frames")
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, EndFrame):
            # Shutting down: let whatever was buffered play out first.
            await self.resume()
            await self.push_frame(frame, direction)
            return

        if self._paused:
            self._buffer.append(frame)
            return

        await self.push_frame(frame, direction)
