"""MetricsObserver: turns Pipecat MetricsFrames and speaking events into MetricsCollector calls."""

from collections import OrderedDict

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    FunctionCallInProgressFrame,
    MetricsFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    STTUsageMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

from tutor.metrics import MetricsCollector


class FrameDeduper:
    """Observers see a frame once per pipeline hop; only count each frame once."""

    def __init__(self, capacity: int = 2048):
        self._seen: OrderedDict[int, None] = OrderedDict()
        self._capacity = capacity

    def first_time(self, frame: Frame) -> bool:
        if frame.id in self._seen:
            return False
        self._seen[frame.id] = None
        if len(self._seen) > self._capacity:
            self._seen.popitem(last=False)
        return True


class MetricsObserver(BaseObserver):
    def __init__(self, collector: MetricsCollector):
        super().__init__()
        self.collector = collector
        self._dedupe = FrameDeduper()
        self._user_stopped_at: int | None = None

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if not self._dedupe.first_time(frame):
            return

        if isinstance(frame, MetricsFrame):
            for item in frame.data:
                if isinstance(item, TTFBMetricsData):
                    self.collector.record_ttfb(item.processor, item.value)
                elif isinstance(item, ProcessingMetricsData):
                    self.collector.record_processing(item.processor, item.value)
                elif isinstance(item, LLMUsageMetricsData):
                    usage = item.value
                    self.collector.record_llm_usage(
                        item.model,
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        usage.total_tokens,
                        usage.cache_read_input_tokens or 0,
                    )
                elif isinstance(item, TTSUsageMetricsData):
                    self.collector.record_tts_usage(item.value)
                elif isinstance(item, STTUsageMetricsData):
                    self.collector.record_stt_usage(item.value.audio_seconds)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_stopped_at = data.timestamp
            self.collector.record_user_turn()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self.collector.record_bot_turn()
            if self._user_stopped_at is not None:
                self.collector.record_turn_latency((data.timestamp - self._user_stopped_at) / 1e9)
                self._user_stopped_at = None
        elif isinstance(frame, FunctionCallInProgressFrame):
            self.collector.record_tool_call(frame.function_name)
