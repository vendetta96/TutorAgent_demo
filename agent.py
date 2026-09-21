"""Tutor Bot pipeline wiring (Pipecat + OpenAI only).

Pipeline:
  transport.in -> STT -> SafetyProcessor -> user aggregator -> ContextAugmenter
    -> LLM -> TTS -> PauseGate -> transport.out -> assistant aggregator

Business logic lives in the framework-free `tutor` package; this module only wires it in.
"""

from dataclasses import dataclass

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAIRealtimeSTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

from tutor.config import Settings
from tutor.controller import PresentationController
from tutor.knowledge.embedder import OpenAIEmbedder
from tutor.knowledge.store import KnowledgeBase, VectorStore
from tutor.metrics import MetricsCollector, render_report
from tutor.processors.context_augmenter import ContextAugmenter
from tutor.processors.metrics_observer import MetricsObserver
from tutor.processors.pause_gate import PauseGate
from tutor.processors.presentation_driver import PresentationDriver
from tutor.processors.safety_processor import PausedUserMuteStrategy, SafetyProcessor
from tutor.prompts import build_system_prompt
from tutor.safety import OpenAIModerator, SafetyVerdict
from tutor.tools import build_tools
from tutor.transcripts import TranscriptRecorder

load_dotenv(override=True)

# Most recent session report, exposed by main.py at GET /report/latest for the UI.
LAST_REPORT: dict = {}

TTS_INSTRUCTIONS = (
    "You are a warm, upbeat teacher speaking to a class of 10 to 14 year olds. "
    "Clear, natural pace, friendly and encouraging."
)


def load_knowledge(settings: Settings) -> KnowledgeBase | None:
    store = VectorStore.load(settings.knowledge_index)
    if len(store) == 0:
        logger.warning(
            f"[knowledge] no index at {settings.knowledge_index}; run `uv run python -m tutor.knowledge.ingest`"
        )
        return None
    embedder = OpenAIEmbedder(api_key=settings.openai_api_key, model=store.model or settings.embedding_model)
    logger.info(f"[knowledge] loaded {len(store)} chunks (model={store.model})")
    return KnowledgeBase(store, embedder, k=settings.retrieval_k, min_score=settings.retrieval_min_score)


@dataclass
class Session:
    task: PipelineTask
    controller: PresentationController
    driver: PresentationDriver
    collector: MetricsCollector
    recorder: TranscriptRecorder
    context: LLMContext

    def finish(self) -> dict:
        self.driver.shutdown()
        self.collector.finish()
        report = self.collector.report()
        report["controller"] = self.controller.stats.__dict__.copy()
        report["session_id"] = self.recorder.session_id
        self.recorder.session_ended(metrics=report, controller_stats=self.controller.stats.__dict__.copy())
        LAST_REPORT.clear()
        LAST_REPORT.update(report)
        return report


def build_session(websocket_client, settings: Settings | None = None) -> Session:
    """Construct the whole pipeline. No network calls happen here."""
    settings = settings or Settings()

    ws_transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=ProtobufFrameSerializer(),
        ),
    )

    stt = OpenAIRealtimeSTTService(
        api_key=settings.openai_api_key,
        settings=OpenAIRealtimeSTTService.Settings(
            model=settings.stt_model,
            prompt="A school lesson about natural disasters: earthquakes, tsunamis, volcanoes, floods, hurricanes.",
        ),
    )
    tts = OpenAITTSService(
        api_key=settings.openai_api_key,
        settings=OpenAITTSService.Settings(
            model=settings.tts_model, voice=settings.tts_voice, instructions=TTS_INSTRUCTIONS
        ),
    )
    llm = OpenAILLMService(
        api_key=settings.openai_api_key,
        settings=OpenAILLMService.Settings(model=settings.llm_model, temperature=0.6),
    )

    # ---- business logic ---------------------------------------------------
    controller = PresentationController(max_idle_prompts=settings.max_idle_prompts)
    collector = MetricsCollector()
    recorder = TranscriptRecorder(settings.transcript_dir)
    gate = PauseGate()
    driver = PresentationDriver(
        controller,
        gate,
        recorder=recorder,
        collector=collector,
        slide_gap_secs=settings.slide_gap_secs,
        stall_secs=settings.stall_secs,
        qna_idle_secs=settings.qna_idle_secs,
    )
    tools = build_tools(controller, driver)
    knowledge = load_knowledge(settings)

    def on_safety_block(text: str, verdict: SafetyVerdict) -> None:
        collector.record_event("safety_block")
        recorder.event(
            "safety_block",
            category=verdict.category.value if verdict.category else None,
            slide=controller.slide_number,
            mode=controller.mode.value,
        )
        # A blocked interruption still needs the slide to be finished afterwards.
        controller.resume_after_reply = controller.interrupted_mid_slide

    def on_retrieval(query: str, passages: list[str]) -> None:
        if passages:
            collector.record_event("knowledge_hits", len(passages))
            recorder.event("retrieval", query=query, passages=len(passages))

    moderator = OpenAIModerator(api_key=settings.openai_api_key).check if settings.use_moderation_api else None
    safety = SafetyProcessor(on_block=on_safety_block, moderator=moderator)
    augmenter = ContextAugmenter(controller, knowledge, on_retrieval=on_retrieval)

    system_prompt = build_system_prompt(learned_guidance=settings.learned_guidance())
    context = LLMContext([{"role": "system", "content": system_prompt}], tools=tools)

    vad_params = VADParams(confidence=0.8, start_secs=0.4, stop_secs=0.4, min_volume=0.6)
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=vad_params),
            user_mute_strategies=[PausedUserMuteStrategy(lambda: controller.paused)],
        ),
    )

    pipeline = Pipeline(
        [
            ws_transport.input(),
            stt,
            safety,
            context_aggregator.user(),
            augmenter,
            llm,
            tts,
            gate,
            ws_transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[driver, MetricsObserver(collector)],
        idle_timeout_secs=900,
    )
    driver.set_task(task)

    # ---- UI <-> agent messaging (RTVI) -------------------------------------
    async def push_state(snapshot: dict) -> None:
        await task.rtvi.send_server_message({"type": "state", **snapshot})

    driver.add_state_listener(push_state)

    @task.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await driver.start()

    @task.rtvi.event_handler("on_client_message")
    async def on_client_message(rtvi, msg):
        if msg.type == "pause":
            changed = await driver.pause()
        elif msg.type == "resume":
            changed = await driver.resume()
        elif msg.type == "state":
            changed = True
            await push_state(controller.snapshot())
        else:
            await rtvi.send_error_response(msg, f"unknown message type {msg.type!r}")
            return
        await rtvi.send_server_response(msg, {"ok": True, "changed": changed, **controller.snapshot()})

    # ---- transcript recording --------------------------------------------
    @context_aggregator.user().event_handler("on_user_turn_message_added")
    async def on_user_turn(aggregator, message):
        recorder.user(
            message.content,
            slide=controller.slide_number,
            mode=controller.mode.value,
            interrupted=controller.interrupted_mid_slide,
        )

    @context_aggregator.assistant().event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn(aggregator, message):
        if message.content:
            recorder.bot(
                message.content,
                slide=controller.slide_number,
                mode=controller.mode.value,
                interrupted=message.interrupted,
            )

    @ws_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("[transport] client connected")
        recorder.session_started(llm=settings.llm_model, stt=settings.stt_model, tts=settings.tts_model)

    @ws_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("[transport] client disconnected")
        await task.cancel()

    return Session(task, controller, driver, collector, recorder, context)


async def run_bot(websocket_client, settings: Settings | None = None):
    session = build_session(websocket_client, settings)
    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(session.task)
    finally:
        report = session.finish()
        stats = session.controller.stats
        print(render_report(report))
        print(
            f"Lesson: slides_presented={stats.slides_presented} user_turns={stats.user_turns} "
            f"interruptions={stats.interruptions} jumps={stats.jumps} pauses={stats.pauses} "
            f"stalls_recovered={stats.stalls_recovered}\n"
            f"Transcript: {session.recorder.path}"
        )
