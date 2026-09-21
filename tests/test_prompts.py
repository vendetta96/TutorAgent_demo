from tutor.controller import PresentationController
from tutor.prompts import (
    KNOWLEDGE_MARKER,
    SLIDE_MARKER,
    STATE_MARKER,
    build_state_note,
    build_system_prompt,
    continue_slide_message,
    enter_qna_message,
    idle_prompt_message,
    is_state_note,
    slide_message,
)
from tutor.slides import DECK


def test_system_prompt_mentions_markers_tools_and_every_slide():
    prompt = build_system_prompt()
    assert SLIDE_MARKER in prompt and STATE_MARKER in prompt and KNOWLEDGE_MARKER in prompt
    for tool in ("go_to_slide", "next_slide", "end_session"):
        assert tool in prompt
    for slide in DECK:
        assert slide.title in prompt
    assert "personal information" in prompt.lower()


def test_learned_guidance_is_appended_only_when_present():
    base = build_system_prompt()
    assert "COACHING NOTES" not in base
    assert "COACHING NOTES" not in build_system_prompt(learned_guidance="   ")
    with_notes = build_system_prompt(learned_guidance="- Students ask about tsunamis on slide 4.")
    assert "COACHING NOTES" in with_notes
    assert "tsunamis on slide 4" in with_notes


def test_slide_messages_carry_slide_content():
    msg = slide_message(DECK[2], first=False)
    assert msg.startswith(SLIDE_MARKER)
    assert "SLIDE 3" in msg and DECK[2].talking_points[0] in msg
    assert "Begin the lesson" in slide_message(DECK[0], first=True)
    assert "Continue from where you left off" in continue_slide_message(DECK[2])
    assert "Q&A" in enter_qna_message() or "questions" in enter_qna_message()
    assert idle_prompt_message(1) != idle_prompt_message(2)


def test_state_note_presenting_fresh():
    c = PresentationController()
    c.start()
    note = build_state_note(c)
    assert note.startswith(STATE_MARKER)
    assert "PRESENTING" in note and "slide: 1 of" in note
    assert "Present this slide now" in note


def test_state_note_interrupted_mid_slide_asks_to_resume_where_stopped():
    c = PresentationController()
    c.start()
    c.on_bot_started_speaking()
    c.on_user_started_speaking()
    note = build_state_note(c)
    assert "interrupted you in the middle" in note
    assert "exactly where you stopped" in note
    assert "Do not restart" in note


def test_state_note_after_slide_finished():
    c = PresentationController()
    c.start()
    c.on_bot_started_speaking()
    c.on_bot_stopped_speaking()
    note = build_state_note(c)
    assert "just finished presenting this slide" in note


def test_state_note_qna_and_ended():
    c = PresentationController()
    c.start()
    c.go_to_slide(len(DECK))
    c.next_slide()
    note = build_state_note(c)
    assert "Mode: Q&A" in note and "go_to_slide" in note and "end_session" in note
    c.end()
    assert "Mode: ENDED" in build_state_note(c)


def test_state_note_includes_knowledge_passages():
    c = PresentationController()
    c.start()
    note = build_state_note(c, ["Tsunamis are caused by undersea earthquakes.", "Second fact."])
    assert KNOWLEDGE_MARKER in note
    assert "(1) Tsunamis" in note and "(2) Second fact." in note


def test_is_state_note_detects_only_our_notes():
    assert is_state_note({"role": "system", "content": f"{STATE_MARKER}\nMode: X"})
    assert not is_state_note({"role": "system", "content": "You are a teacher"})
    assert not is_state_note({"role": "user", "content": STATE_MARKER})
    assert not is_state_note({"role": "system", "content": [{"type": "text", "text": STATE_MARKER}]})
