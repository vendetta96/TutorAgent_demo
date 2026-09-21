"""Constructs the real Pipecat pipeline (no network) to catch wiring mistakes."""

from pathlib import Path
from unittest.mock import MagicMock

from pipecat.processors.frameworks.rtvi import RTVIProcessor

from agent import build_session
from tutor.config import Settings
from tutor.controller import Mode
from tutor.prompts import STATE_MARKER


def flatten(processor) -> list:
    inner = getattr(processor, "processors", None)
    if not inner:
        return [processor]
    return [p for child in inner for p in flatten(child)]


def make_settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.openai_api_key = "sk-test-not-real"
    s.transcript_dir = tmp_path / "transcripts"
    s.knowledge_index = tmp_path / "missing_index.json"
    s.learned_guidance_path = tmp_path / "missing_guidance.md"
    return s


def test_pipeline_builds_with_all_processors_and_tools(tmp_path):
    session = build_session(MagicMock(), make_settings(tmp_path))
    names = [type(p).__name__ for p in flatten(session.task.pipeline)]
    names = [n for n in names if n not in {"PipelineSource", "PipelineSink"}]
    assert names == [
        "RTVIProcessor",
        "FastAPIWebsocketInputTransport",
        "OpenAIRealtimeSTTService",
        "SafetyProcessor",
        "LLMUserAggregator",
        "ContextAugmenter",
        "OpenAILLMService",
        "OpenAITTSService",
        "PauseGate",
        "FastAPIWebsocketOutputTransport",
        "LLMAssistantAggregator",
    ]
    assert isinstance(session.task.rtvi, RTVIProcessor)
    tools = session.context.tools
    assert sorted(s.name for s in tools.standard_tools) == ["end_session", "go_to_slide", "next_slide"]
    assert all(s.handler is not None for s in tools.standard_tools)
    assert session.context.messages[0]["role"] == "system"
    assert STATE_MARKER in session.context.messages[0]["content"]
    assert session.controller.mode is Mode.IDLE


def test_learned_guidance_is_loaded_into_the_system_prompt(tmp_path):
    settings = make_settings(tmp_path)
    settings.learned_guidance_path.write_text("- Mention tsunamis on slide 4.")
    session = build_session(MagicMock(), settings)
    assert "Mention tsunamis on slide 4." in session.context.messages[0]["content"]


def test_finish_produces_report_and_transcript(tmp_path):
    session = build_session(MagicMock(), make_settings(tmp_path))
    session.recorder.session_started()
    report = session.finish()
    assert report["session_id"] == session.recorder.session_id
    assert "controller" in report and report["controller"]["slides_presented"] == 0
    assert session.recorder.path.exists()
