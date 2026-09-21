"""Driver tests: a fake PipelineTask records injected frames; frames are fed as observer events."""

import asyncio

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
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
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection

from tutor.controller import Mode, PresentationController
from tutor.metrics import MetricsCollector
from tutor.processors.pause_gate import PauseGate
from tutor.processors.presentation_driver import PresentationDriver
from tutor.prompts import SLIDE_MARKER
from tutor.slides import DECK


class FakeTask:
    def __init__(self):
        self.frames: list[Frame] = []
        self.stopped = False

    async def queue_frames(self, frames):
        self.frames.extend(frames)

    async def stop_when_done(self):
        self.stopped = True

    def injected(self) -> list[str]:
        return [
            f.messages[0]["content"]
            for f in self.frames
            if isinstance(f, LLMMessagesAppendFrame)
        ]


@pytest.fixture
def setup(monkeypatch):
    controller = PresentationController(max_idle_prompts=1)
    gate = PauseGate()

    async def fake_push(frame, direction=FrameDirection.DOWNSTREAM):
        pass

    monkeypatch.setattr(gate, "push_frame", fake_push)
    collector = MetricsCollector()
    driver = PresentationDriver(
        controller,
        gate,
        collector=collector,
        slide_gap_secs=0.05,
        stall_secs=0.3,
        qna_idle_secs=0.3,
        end_grace_secs=0.5,
        start_fallback_secs=0.1,
        watchdog_tick_secs=0.05,
    )
    task = FakeTask()
    driver.set_task(task)
    yield controller, driver, task, gate, collector
    driver.shutdown()


async def push(driver: PresentationDriver, frame: Frame):
    await driver.on_push_frame(FramePushed(source=None, destination=None, frame=frame, direction=FrameDirection.DOWNSTREAM, timestamp=0))  # type: ignore[arg-type]


async def speak_cycle(driver: PresentationDriver):
    await push(driver, LLMFullResponseStartFrame())
    await push(driver, BotStartedSpeakingFrame())
    await push(driver, LLMFullResponseEndFrame())
    await push(driver, BotStoppedSpeakingFrame())


async def test_start_injects_first_slide_once(setup):
    controller, driver, task, *_ = setup
    await driver.start()
    await driver.start()
    assert len(task.injected()) == 1
    assert task.injected()[0].startswith(SLIDE_MARKER) and "SLIDE 1" in task.injected()[0]
    assert controller.mode is Mode.PRESENTING


async def test_start_fallback_fires_after_start_frame(setup):
    controller, driver, task, *_ = setup
    await push(driver, StartFrame())
    assert task.injected() == []
    await asyncio.sleep(0.2)
    assert len(task.injected()) == 1


async def test_bot_finishing_a_slide_advances_after_the_gap(setup):
    controller, driver, task, *_ = setup
    await driver.start()
    await speak_cycle(driver)
    assert len(task.injected()) == 1  # not yet: gap running
    await asyncio.sleep(0.12)
    assert len(task.injected()) == 2 and "SLIDE 2" in task.injected()[1]
    assert controller.slide_number == 2


async def test_frames_seen_on_multiple_hops_are_counted_once(setup):
    controller, driver, task, *_ = setup
    await driver.start()
    frame = BotStoppedSpeakingFrame()
    await push(driver, BotStartedSpeakingFrame())
    await push(driver, frame)
    await push(driver, frame)
    await asyncio.sleep(0.12)
    assert controller.slide_number == 2


async def test_user_speaking_cancels_the_advance(setup):
    controller, driver, task, *_ = setup
    await driver.start()
    await speak_cycle(driver)
    await push(driver, UserStartedSpeakingFrame())
    await asyncio.sleep(0.12)
    assert controller.slide_number == 1
    await push(driver, UserStoppedSpeakingFrame())
    await speak_cycle(driver)  # bot answers, then the gap advances
    await asyncio.sleep(0.12)
    assert controller.slide_number == 2


async def test_advance_waits_while_llm_or_tool_call_is_busy(setup):
    controller, driver, task, *_ = setup
    await driver.start()
    await push(driver, BotStartedSpeakingFrame())
    await push(driver, LLMFullResponseStartFrame())
    await push(driver, BotStoppedSpeakingFrame())
    await asyncio.sleep(0.12)
    assert controller.slide_number == 1  # LLM still generating
    await push(driver, LLMFullResponseEndFrame())
    await asyncio.sleep(0.12)
    assert controller.slide_number == 2

    await push(driver, BotStartedSpeakingFrame())
    await push(driver, FunctionCallsStartedFrame(function_calls=[object()]))
    await push(driver, BotStoppedSpeakingFrame())
    await asyncio.sleep(0.12)
    assert controller.slide_number == 2  # tool call pending
    await push(driver, FunctionCallResultFrame(function_name="x", tool_call_id="1", arguments={}, result={}))
    await asyncio.sleep(0.12)
    assert controller.slide_number == 3


async def test_pause_blocks_advance_and_resume_continues(setup):
    controller, driver, task, gate, collector = setup
    await driver.start()
    await speak_cycle(driver)
    assert await driver.pause() is True
    assert gate.paused and controller.paused
    await asyncio.sleep(0.12)
    assert controller.slide_number == 1
    assert await driver.pause() is False
    assert await driver.resume() is True
    assert not gate.paused
    await asyncio.sleep(0.12)
    assert controller.slide_number == 2
    assert collector.events["pause"] == 1
    assert await driver.resume() is False


async def test_full_deck_enters_qna_then_idle_prompts_then_ends(setup):
    controller, driver, task, *_ = setup
    await driver.start()
    for _ in range(len(DECK)):
        await speak_cycle(driver)
        await asyncio.sleep(0.12)
    assert controller.mode is Mode.QNA
    assert "presentation is complete" in task.injected()[-1]
    await speak_cycle(driver)  # bot invites questions
    await asyncio.sleep(0.5)  # idle -> prompt
    assert "Nobody has said anything" in task.injected()[-1]
    await speak_cycle(driver)
    await asyncio.sleep(0.5)  # idle again -> end session (max_idle_prompts=1)
    assert controller.mode is Mode.ENDED
    assert "goodbye" in task.injected()[-1].lower()
    assert not task.stopped
    await speak_cycle(driver)  # goodbye spoken
    assert task.stopped


async def test_end_session_via_tool_does_not_inject_a_second_goodbye(setup):
    controller, driver, task, *_ = setup
    await driver.start()
    before = len(task.injected())
    await driver.begin_ending(inject_goodbye=False)
    assert len(task.injected()) == before
    assert controller.mode is Mode.ENDED
    await speak_cycle(driver)
    assert task.stopped


async def test_end_grace_timer_ends_session_if_bot_never_speaks(setup):
    controller, driver, task, *_ = setup
    await driver.start()
    await driver.begin_ending(inject_goodbye=True)
    await asyncio.sleep(0.7)
    assert task.stopped


async def test_stall_watchdog_nudges_when_nothing_happens(setup):
    controller, driver, task, *_ = setup
    await push(driver, StartFrame())
    await driver.start()
    await asyncio.sleep(0.6)  # no bot speech at all -> re-present slide 1
    assert controller.stats.stalls_recovered >= 1
    assert "SLIDE 1" in task.injected()[-1]


async def test_state_listener_receives_snapshots(setup):
    controller, driver, task, *_ = setup
    snapshots = []

    async def listener(snap):
        snapshots.append(snap)

    driver.add_state_listener(listener)
    await driver.start()
    await push(driver, BotStartedSpeakingFrame())
    assert snapshots[-1]["bot_speaking"] is True and snapshots[-1]["slide_number"] == 1


async def test_cancel_frame_shuts_down_timers(setup):
    controller, driver, task, *_ = setup
    await push(driver, StartFrame())
    await driver.start()
    await speak_cycle(driver)
    await push(driver, CancelFrame())
    await asyncio.sleep(0.12)
    assert controller.slide_number == 1  # advance timer was cancelled
