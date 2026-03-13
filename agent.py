import asyncio
from typing import List
import os

from pipecat.observers.base_observer import BaseObserver, FramePushed
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    LLMMessagesAppendFrame,
    UserStartedSpeakingFrame, StartFrame,
)
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

load_dotenv(override=True)


SLIDE_SYSTEM_MESSAGES: List[str] = [
    # Slide 1 – welcome & overview
    (
        "SLIDE 1: WELCOME & OVERVIEW\n\n"
        "Welcome the audience and briefly introduce the topic: Natural Disasters. "
        "Explain that this presentation will walk through what natural disasters are, why they occur, "
        "and how they affect people and the environment. "
        "Mention that questions are welcome at any time and that you will continue guiding them through the slides."
    ),
    # Slide 2 – what are natural disasters
    (
        "SLIDE 2: WHAT ARE NATURAL DISASTERS\n\n"
        "Explain that natural disasters are extreme natural events that cause major damage to life, property, "
        "or the environment. Examples include earthquakes, floods, hurricanes, volcanic eruptions, and droughts. "
        "Emphasize that these events are caused by natural processes of the Earth."
    ),
    # Slide 3 – why they happen
    (
        "SLIDE 3: WHY NATURAL DISASTERS HAPPEN\n\n"
        "Describe the main reasons natural disasters occur: movement of tectonic plates, extreme weather patterns, "
        "volcanic activity, and climate-related changes. "
        "Briefly mention that some disasters are sudden while others develop slowly over time."
    ),
    # Slide 4 – major types
    (
        "SLIDE 4: MAJOR TYPES OF NATURAL DISASTERS\n\n"
        "Introduce the most common categories such as earthquakes, floods, cyclones, wildfires, landslides, "
        "and volcanic eruptions. "
        "Explain that each type has different causes and impacts depending on geography and climate."
    ),
    # Slide 5 – impacts on people
    (
        "SLIDE 5: IMPACT ON PEOPLE\n\n"
        "Explain how natural disasters affect communities: loss of life, injuries, destruction of homes, "
        "and displacement of families. "
        "Also mention disruption to healthcare, education, and daily life."
    ),
    # Slide 6 – environmental effects
    (
        "SLIDE 6: ENVIRONMENTAL EFFECTS\n\n"
        "Describe how natural disasters affect ecosystems: deforestation from wildfires, flooding of habitats, "
        "soil erosion, and pollution of water sources. "
        "Mention that while disasters cause destruction, some also reshape landscapes and ecosystems."
    ),
    # Slide 7 – preparedness and safety
    (
        "SLIDE 7: PREPAREDNESS AND SAFETY\n\n"
        "Explain how preparation can reduce damage and save lives. "
        "Discuss early warning systems, evacuation plans, emergency kits, and community awareness. "
        "Highlight that education and planning are key to disaster resilience."
    ),
    # Slide 8 – conclusion & discussion
    (
        "SLIDE 8: CONCLUSION & DISCUSSION\n\n"
        "Summarize that natural disasters are powerful natural events that can have serious impacts on society "
        "and the environment. "
        "Emphasize the importance of preparedness, scientific understanding, and community cooperation. "
        "Invite the audience to ask questions or request clarification on any slide."
    ),
]

class PresentationObserver0(BaseObserver):
    """Observer that advances slides after 5 seconds of bot silence."""

    def __init__(self):
        super().__init__()
        self.current_slide = -1
        self.task: PipelineTask | None = None
        self._is_bot_speaking = False
        self._silence_timer: asyncio.TimerHandle | None = None
        self._user_spoke_since_last_slide = False

    def set_task(self, task: PipelineTask):
        self.task = task

    def _cancel_silence_timer(self):
        if self._silence_timer:
            self._silence_timer.cancel()
            self._silence_timer = None

    def _schedule_silence_check(self):
        # Schedule a check 5 seconds after the bot stops speaking.
        self._cancel_silence_timer()
        loop = asyncio.get_event_loop()
        self._silence_timer = loop.call_later(
            3.0,
            lambda: asyncio.create_task(self._on_silence_timeout()),
        )

    async def _on_silence_timeout(self):
        if not self._is_bot_speaking:
            if self._user_spoke_since_last_slide:
                logger.info("3 seconds silence after user spoke; staying on slide and instructing AI to continue.")
                await self.continue_current_slide()
            else:
                logger.info("3 seconds of bot silence detected; queuing next slide.")
                await self.go_to_next_slide()

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame

        if isinstance(frame, StartFrame):
            # Pipeline just started, force start with first slide
            await self._on_silence_timeout()

        elif isinstance(frame, BotStartedSpeakingFrame):
            self._is_bot_speaking = True
            self._cancel_silence_timer()

        elif isinstance(frame, UserStartedSpeakingFrame):
            self._user_spoke_since_last_slide = True
            self._cancel_silence_timer()

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._is_bot_speaking = False
            self._schedule_silence_check()

        elif isinstance(frame, (EndFrame, CancelFrame)):
            # Pipeline is ending; stop any pending timers.
            self._cancel_silence_timer()

        # Observers are side-effect-only; nothing to push downstream.

    async def continue_current_slide(self):
        """Instruct the AI to stay on the current slide and continue where it left off."""
        if self.current_slide < 0 or self.current_slide >= len(SLIDE_SYSTEM_MESSAGES):
            return
        self._user_spoke_since_last_slide = False
        slide_num = self.current_slide + 1
        slide_content = SLIDE_SYSTEM_MESSAGES[self.current_slide]
        slide_title = slide_content.split("\n\n")[0].strip() if slide_content else f"Slide {slide_num}"
        new_messages = [
            {
                "role": "system",
                "content": (
                    f"You are still on {slide_title}. Continue presenting this slide where you left off. "
                    "Do not repeat what you already said; pick up from there."
                ),
            }
        ]
        await self.task.queue_frames([LLMMessagesAppendFrame(messages=new_messages, run_llm=True)])

    async def go_to_next_slide(self):
        self.current_slide += 1
        self._user_spoke_since_last_slide = False
        new_messages = []
        if self.current_slide < len(SLIDE_SYSTEM_MESSAGES) - 1:
            print(f"Adding slide {self.current_slide} to context")
            new_messages.append({"role": "system", "content": SLIDE_SYSTEM_MESSAGES[self.current_slide]})
        elif self.current_slide == len(SLIDE_SYSTEM_MESSAGES) - 1:
            print(f"Adding goodbye slide to context")
            new_messages.append({"role": "system", "content": "Say goodbye and end the presentation."})

        if len(new_messages) > 0:
            await self.task.queue_frames([LLMMessagesAppendFrame(messages=new_messages, run_llm=True)])
        else:
            logger.critical("NO SLIDE TO INSERT")


async def run_bot(websocket_client):
    ws_transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=ProtobufFrameSerializer(),
        ),
    )

    messages = []

    stt = OpenAIRealtimeSTTService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-transcribe",
    )

    tts = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini-tts",
        voice="alloy",
        instructions="AI presenter for business people. Speak fast.",
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",
    )

    context = LLMContext(messages)

    # Stricter VAD to reduce false "user spoke" from background noise: higher confidence,
    # longer sustained speech before trigger, higher minimum volume.
    vad_params = VADParams(
        confidence=0.85,
        start_secs=0.45,
        stop_secs=0.35,
        min_volume=0.7,
    )
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=vad_params),
        ),
    )

    pipeline = Pipeline(
        [
            ws_transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            ws_transport.output(),
            context_aggregator.assistant(),
        ]
    )

    presentation_observer0 = PresentationObserver0()
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[presentation_observer0],
        enable_turn_tracking=False
    )
    presentation_observer0.set_task(task)

    @ws_transport.event_handler("on_client_connected")
    async def on_client_connected():
        logger.info("[transport] client connected")

    @ws_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected():
        logger.info("[transport] client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
