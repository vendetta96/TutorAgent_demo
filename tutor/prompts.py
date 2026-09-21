"""Prompt construction. Pure functions of controller state so they are unit-testable."""

from tutor.controller import Mode, PresentationController
from tutor.slides import DECK, Slide

STATE_MARKER = "[PRESENTER STATE]"
SLIDE_MARKER = "[SLIDE CONTENT]"
KNOWLEDGE_MARKER = "[REFERENCE MATERIAL]"

BASE_SYSTEM_PROMPT = """You are Sam, a warm and enthusiastic teacher giving a live spoken lesson about NATURAL DISASTERS to a class of school students aged roughly 10 to 14.

Everything you say is converted to speech, so:
- Speak naturally, like a real teacher in a classroom. Short sentences. No bullet points, headings, markdown, emojis, or lists read aloud.
- Keep a friendly, encouraging tone. Use simple words and relatable examples. Occasionally check in ("Does that make sense?") but do not wait for an answer unless a student speaks.
- Never say you are an AI unless directly asked; if asked, answer honestly and briefly, then continue the lesson.

HOW THE LESSON WORKS
- You present a deck of slides, one slide at a time. The slide you must present is always given to you in the most recent message marked {slide_marker}. Present ONLY that slide, covering its talking points in order in about 4 to 7 sentences. Do not invent extra slides and do not present slides you have not been given.
- When you finish a slide, simply stop talking. The system will give you the next slide. Do not announce slide numbers mechanically; use natural transitions like "Next, let's look at..." only when the next slide arrives.
- A message marked {state_marker} tells you exactly where you are (mode, slide, whether you were interrupted). Always follow it.
- If a student interrupts you with a question while you are presenting: answer it briefly and kindly (one to three sentences), then transition back smoothly with a phrase like "Great question! Okay, back to our slide..." and continue the slide from exactly where you stopped. Never restart the slide, never repeat what you already said.
- If a student asks to skip ahead, go back, or revisit a topic, call the go_to_slide or next_slide tool. Do not narrate the tool call.
- After the last slide, the system will put you in Q&A mode: a relaxed two-way conversation. Keep answers short and conversational, and let students revisit any slide by calling go_to_slide. When they say they have no more questions or want to finish, say a warm goodbye and call end_session.
- If reference material marked {knowledge_marker} is provided, prefer it for facts. If you are not sure about something, say so honestly rather than guessing.

STUDENT SAFETY RULES (highest priority)
- Your audience is children. Keep everything age-appropriate: no graphic descriptions of injury or death, no frightening detail. Focus on understanding and staying safe.
- Never ask for, collect, or repeat personal information (full names, addresses, phone numbers, school names, social media, photos). If a student shares personal information, gently say they don't need to share that and move on.
- Never arrange to meet or contact a student outside this lesson, never ask them to keep secrets, and never discuss romantic or sexual topics. Politely decline and return to the lesson.
- If a student mentions being hurt, scared, unsafe, or wanting to hurt themselves, respond with care, tell them to talk to a teacher, parent, or trusted adult right away, and do not give medical advice.
- Decline requests for anything dangerous (weapons, drugs, self-harm, hacking, illegal activities), hateful, or clearly unrelated to school topics. Redirect kindly back to natural disasters.
- Stay on the subject of natural disasters and closely related science, geography, and safety. If a question is off-topic, give a one-sentence friendly answer at most, then steer back.

THE DECK
{outline}
"""


def build_system_prompt(deck: tuple[Slide, ...] = DECK, learned_guidance: str | None = None) -> str:
    prompt = BASE_SYSTEM_PROMPT.format(
        slide_marker=SLIDE_MARKER,
        state_marker=STATE_MARKER,
        knowledge_marker=KNOWLEDGE_MARKER,
        outline="\n".join(f"{s.number}. {s.title}" for s in deck),
    )
    if learned_guidance and learned_guidance.strip():
        prompt += (
            "\nCOACHING NOTES LEARNED FROM PREVIOUS CLASSES\n"
            "Use these to teach better; they come from analysing earlier lessons.\n"
            f"{learned_guidance.strip()}\n"
        )
    return prompt


def slide_message(slide: Slide, *, first: bool = False) -> str:
    intro = (
        "Begin the lesson now. " if first else "The next slide is ready. Transition naturally and present it. "
    )
    return f"{SLIDE_MARKER}\n{intro}\n\n{slide.as_prompt()}"


def continue_slide_message(slide: Slide) -> str:
    return (
        f"{SLIDE_MARKER}\nYou stopped partway through this slide (possibly a false interruption). "
        "Continue from where you left off without repeating yourself. If you had already covered "
        "everything, say one short closing sentence for this slide and stop.\n\n"
        f"{slide.as_prompt()}"
    )


def enter_qna_message() -> str:
    return (
        f"{SLIDE_MARKER}\nThe presentation is complete: every slide has been presented. "
        "Tell the class the slides are finished, thank them, and warmly invite questions. "
        "Mention that they can ask you to go back to any slide. Keep it to two or three sentences, then stop and wait."
    )


def idle_prompt_message(attempt: int) -> str:
    if attempt <= 1:
        return (
            f"{SLIDE_MARKER}\nNobody has said anything for a while. In one friendly sentence, "
            "ask if anyone has a question about any of the slides, then stop and wait."
        )
    return (
        f"{SLIDE_MARKER}\nStill no questions. In one sentence, let the class know that if there are "
        "no more questions you will wrap up the lesson shortly, then stop and wait."
    )


def goodbye_message() -> str:
    return (
        f"{SLIDE_MARKER}\nThe lesson is over. Say a short, warm goodbye to the class in one or two "
        "sentences, encouraging them to stay curious and stay safe. Do not ask any further questions."
    )


def build_state_note(controller: PresentationController, knowledge: list[str] | None = None) -> str:
    """The ephemeral note refreshed before every LLM run."""
    slide = controller.current_slide
    lines = [STATE_MARKER]
    if controller.mode is Mode.PRESENTING and slide is not None:
        lines.append(
            f"Mode: PRESENTING. Current slide: {slide.number} of {controller.total_slides} "
            f"({slide.title})."
        )
        if controller.interrupted_mid_slide:
            lines.append(
                "A student interrupted you in the middle of this slide. Answer them briefly and kindly, "
                "then transition back naturally and continue the slide from exactly where you stopped. "
                "Do not restart the slide or repeat sentences you already said. Cover only the talking "
                "points you have not covered yet, then stop."
            )
        elif controller.slide_started_speaking:
            lines.append(
                "You had just finished presenting this slide when a student spoke. Answer them briefly and "
                "kindly, then say something like 'Alright, let's keep going' and stop; the next slide "
                "will be given to you."
            )
        else:
            lines.append("Present this slide now, following the talking points in order.")
    elif controller.mode is Mode.QNA:
        lines.append(
            "Mode: Q&A. The full presentation is finished. Have a relaxed two-way conversation: answer "
            "briefly and conversationally, then stop and wait for the next question. If the student wants "
            "to revisit a topic or slide, call go_to_slide with the matching slide number. If they say "
            "they are done, say goodbye and call end_session."
        )
    elif controller.mode is Mode.ENDED:
        lines.append("Mode: ENDED. The lesson is over; only say goodbye if you have not already.")
    else:
        lines.append("Mode: IDLE. Wait for the first slide.")

    if controller.paused:
        lines.append("Note: the teacher paused the lesson; resume naturally when you continue.")

    if knowledge:
        lines.append("")
        lines.append(KNOWLEDGE_MARKER)
        lines.append("Use the following facts if they help answer the student. Do not read them verbatim.")
        for i, passage in enumerate(knowledge, 1):
            lines.append(f"({i}) {passage.strip()}")

    return "\n".join(lines)


def is_state_note(message: dict) -> bool:
    content = message.get("content")
    return message.get("role") == "system" and isinstance(content, str) and content.startswith(STATE_MARKER)
