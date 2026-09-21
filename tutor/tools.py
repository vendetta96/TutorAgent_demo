"""LLM tools for navigating the deck and ending the session."""

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from tutor.controller import ActionKind, PresentationController
from tutor.processors.presentation_driver import PresentationDriver
from tutor.slides import Slide


def _slide_payload(slide: Slide, instruction: str) -> dict:
    return {
        "slide_number": slide.number,
        "title": slide.title,
        "talking_points": list(slide.talking_points),
        "instructions": instruction,
    }


def build_tools(controller: PresentationController, driver: PresentationDriver) -> ToolsSchema:
    async def go_to_slide(params: FunctionCallParams):
        raw = params.arguments.get("slide_number")
        topic = params.arguments.get("topic")
        slide: Slide | None = None
        try:
            if raw is not None:
                slide = controller.deck[int(raw) - 1] if 1 <= int(raw) <= controller.total_slides else None
        except (TypeError, ValueError):
            slide = None
        if slide is None and topic:
            slide = controller.find_slide_by_topic(str(topic))
        if slide is None:
            await params.result_callback(
                {"error": f"No such slide. Valid slides are 1 to {controller.total_slides}."}
            )
            return
        action = controller.go_to_slide(slide.number)
        driver._cancel_advance()
        logger.info(f"[tools] go_to_slide -> {slide.number} ({action.reason})")
        if driver.recorder:
            driver.recorder.event("tool", name="go_to_slide", slide=slide.number)
        await params.result_callback(
            _slide_payload(
                slide,
                "Present this slide now in full, in a natural spoken style. When you finish, stop; "
                "the system will continue from here.",
            )
        )

    async def next_slide(params: FunctionCallParams):
        action = controller.next_slide()
        driver._cancel_advance()
        if driver.recorder:
            driver.recorder.event("tool", name="next_slide", slide=controller.slide_number)
        if action.kind is ActionKind.ENTER_QNA:
            await params.result_callback(
                {
                    "presentation_complete": True,
                    "instructions": "That was the last slide. Tell the class the slides are finished and "
                    "invite questions; you are now in Q&A mode.",
                }
            )
            return
        assert action.slide is not None
        await params.result_callback(
            _slide_payload(action.slide, "Present this slide now in full. When you finish, stop.")
        )

    async def end_session(params: FunctionCallParams):
        if driver.recorder:
            driver.recorder.event("tool", name="end_session")
        await driver.begin_ending(inject_goodbye=False)
        await params.result_callback(
            {"instructions": "Say a short, warm goodbye to the class now. Do not ask any more questions."}
        )

    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="go_to_slide",
                description=(
                    "Jump to a specific slide of the natural disasters deck when a student asks to go back, "
                    "skip ahead, repeat, or revisit a topic. Provide the slide number, or a topic if unsure."
                ),
                properties={
                    "slide_number": {
                        "type": "integer",
                        "description": f"Slide number between 1 and {controller.total_slides}.",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Topic words the student mentioned, e.g. 'preparedness' or 'types'.",
                    },
                },
                required=[],
                handler=go_to_slide,
            ),
            FunctionSchema(
                name="next_slide",
                description="Move to the next slide when a student explicitly asks to move on or skip ahead.",
                properties={},
                required=[],
                handler=next_slide,
            ),
            FunctionSchema(
                name="end_session",
                description=(
                    "End the lesson when the class says they have no more questions or want to finish. "
                    "Call it, then say goodbye."
                ),
                properties={},
                required=[],
                handler=end_session,
            ),
        ]
    )
