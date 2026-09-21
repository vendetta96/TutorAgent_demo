import pytest
from pipecat.frames.frames import (
    BotSpeakingFrame,
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from tutor.processors.pause_gate import PauseGate


@pytest.fixture
def gate(monkeypatch):
    g = PauseGate()
    pushed: list[tuple[Frame, FrameDirection]] = []

    async def fake_push(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append((frame, direction))

    monkeypatch.setattr(g, "push_frame", fake_push)
    g.pushed = pushed  # type: ignore[attr-defined]
    return g


def audio(n: int = 1) -> TTSAudioRawFrame:
    return TTSAudioRawFrame(audio=b"\x00" * 320 * n, sample_rate=16000, num_channels=1)


async def test_frames_pass_through_when_not_paused(gate):
    frames = [TTSStartedFrame(), audio(), TextFrame(text="hi"), TTSStoppedFrame()]
    for f in frames:
        await gate.process_frame(f, FrameDirection.DOWNSTREAM)
    assert [f for f, _ in gate.pushed] == frames
    assert gate.buffered == 0


async def test_pause_buffers_bot_output_and_resume_releases_in_order(gate):
    await gate.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
    await gate.pause()
    assert gate.paused
    held = [audio(1), TextFrame(text="one"), audio(2), TextFrame(text="two"), LLMFullResponseEndFrame(), TTSStoppedFrame()]
    for f in held:
        await gate.process_frame(f, FrameDirection.DOWNSTREAM)
    assert gate.buffered == len(held)
    assert len(gate.pushed) == 1  # only the frame before the pause

    await gate.resume()
    assert not gate.paused and gate.buffered == 0
    assert [f for f, _ in gate.pushed][1:] == held  # exact order preserved


async def test_system_and_upstream_frames_bypass_the_gate_while_paused(gate):
    await gate.pause()
    sys_frame = BotSpeakingFrame()
    await gate.process_frame(sys_frame, FrameDirection.DOWNSTREAM)
    up = TextFrame(text="up")
    await gate.process_frame(up, FrameDirection.UPSTREAM)
    assert [(f, d) for f, d in gate.pushed] == [(sys_frame, FrameDirection.DOWNSTREAM), (up, FrameDirection.UPSTREAM)]
    assert gate.buffered == 0


async def test_end_frame_flushes_buffer_then_passes(gate):
    await gate.pause()
    a = audio()
    await gate.process_frame(a, FrameDirection.DOWNSTREAM)
    end = EndFrame()
    await gate.process_frame(end, FrameDirection.DOWNSTREAM)
    assert [f for f, _ in gate.pushed] == [a, end]
    assert not gate.paused


async def test_pause_and_resume_are_idempotent(gate):
    await gate.resume()  # not paused: no-op
    await gate.pause()
    await gate.pause()
    await gate.process_frame(audio(), FrameDirection.DOWNSTREAM)
    await gate.resume()
    await gate.resume()
    assert gate.buffered == 0 and len(gate.pushed) == 1


async def test_clear_drops_buffered_frames(gate):
    await gate.pause()
    await gate.process_frame(audio(), FrameDirection.DOWNSTREAM)
    await gate.process_frame(audio(), FrameDirection.DOWNSTREAM)
    assert gate.clear() == 2
    await gate.resume()
    assert gate.pushed == []
