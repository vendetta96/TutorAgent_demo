import pytest

from tutor.controller import ActionKind, Mode, PresentationController
from tutor.slides import DECK


def run_full_deck(c: PresentationController) -> list[ActionKind]:
    """Simulate: present slide, bot speaks, bot stops, gap elapses, advance."""
    kinds = []
    action = c.start()
    kinds.append(action.kind)
    for _ in range(len(DECK)):
        c.on_bot_started_speaking()
        scheduled = c.on_bot_stopped_speaking()
        assert scheduled is not None and scheduled.kind is ActionKind.SCHEDULE_ADVANCE
        action = c.advance()
        assert action is not None
        kinds.append(action.kind)
    return kinds


def test_start_presents_first_slide():
    c = PresentationController()
    action = c.start()
    assert action.kind is ActionKind.PRESENT_SLIDE
    assert action.slide is not None and action.slide.number == 1
    assert c.mode is Mode.PRESENTING
    assert c.slide_number == 1


def test_deck_advances_deterministically_and_enters_qna_after_last_slide():
    c = PresentationController()
    kinds = run_full_deck(c)
    assert kinds[0] is ActionKind.PRESENT_SLIDE
    assert kinds[1 : len(DECK)] == [ActionKind.PRESENT_SLIDE] * (len(DECK) - 1)
    assert kinds[-1] is ActionKind.ENTER_QNA
    assert c.mode is Mode.QNA
    assert c.completed_slides == set(range(1, len(DECK) + 1))
    assert c.stats.slides_presented == len(DECK)


def test_no_advance_while_user_is_speaking_or_paused():
    c = PresentationController()
    c.start()
    c.on_bot_started_speaking()
    c.on_user_started_speaking()
    assert c.on_bot_stopped_speaking() is None
    c.on_user_stopped_speaking()
    c.pause()
    assert c.advance() is None
    assert c.slide_number == 1


def test_interruption_mid_slide_is_tracked_and_cleared_on_advance():
    c = PresentationController()
    c.start()
    c.on_bot_started_speaking()
    c.on_user_started_speaking()
    assert c.interrupted_mid_slide is True
    assert c.stats.interruptions == 1
    c.on_user_stopped_speaking()
    c.on_bot_stopped_speaking()
    action = c.advance()
    assert action is not None and action.slide.number == 2
    assert c.interrupted_mid_slide is False


def test_user_speaking_after_slide_finished_is_not_an_interruption():
    c = PresentationController()
    c.start()
    c.on_bot_started_speaking()
    c.on_bot_stopped_speaking()
    c.on_user_started_speaking()
    assert c.interrupted_mid_slide is False
    assert c.stats.interruptions == 0


def test_safety_reply_after_interruption_continues_the_slide_before_advancing():
    c = PresentationController()
    c.start()
    c.on_bot_started_speaking()
    c.on_user_started_speaking()
    c.on_user_stopped_speaking()
    c.resume_after_reply = c.interrupted_mid_slide
    c.on_bot_stopped_speaking()  # the fixed safety reply finished
    action = c.advance()
    assert action is not None and action.kind is ActionKind.CONTINUE_SLIDE
    assert action.slide.number == 1
    # Next gap moves on normally.
    c.on_bot_started_speaking()
    c.on_bot_stopped_speaking()
    action = c.advance()
    assert action.kind is ActionKind.PRESENT_SLIDE and action.slide.number == 2


def test_go_to_slide_from_qna_returns_to_presenting_and_finishes_again():
    c = PresentationController()
    run_full_deck(c)
    assert c.mode is Mode.QNA
    action = c.go_to_slide(4)
    assert action.kind is ActionKind.PRESENT_SLIDE and action.slide.number == 4
    assert c.mode is Mode.PRESENTING
    assert c.stats.jumps == 1
    # Presentation resumes from slide 4 through to the end, then Q&A again.
    kinds = []
    for _ in range(len(DECK) - 3):
        c.on_bot_started_speaking()
        c.on_bot_stopped_speaking()
        kinds.append(c.advance().kind)
    assert kinds[-1] is ActionKind.ENTER_QNA
    assert c.mode is Mode.QNA


def test_go_to_slide_validates_range():
    c = PresentationController()
    c.start()
    with pytest.raises(ValueError):
        c.go_to_slide(0)
    with pytest.raises(ValueError):
        c.go_to_slide(len(DECK) + 1)


def test_next_slide_on_last_slide_enters_qna():
    c = PresentationController()
    c.start()
    c.go_to_slide(len(DECK))
    action = c.next_slide()
    assert action.kind is ActionKind.ENTER_QNA
    assert c.mode is Mode.QNA


def test_next_slide_before_start_presents_first():
    c = PresentationController()
    action = c.next_slide()
    assert action.kind is ActionKind.PRESENT_SLIDE and action.slide.number == 1


def test_find_slide_by_topic():
    c = PresentationController()
    assert c.find_slide_by_topic("can we go back to preparedness and safety?").number == 7
    assert c.find_slide_by_topic("tell me about earthquakes and volcanoes again").number == 4
    assert c.find_slide_by_topic("something completely unrelated") is None


def test_pause_and_resume():
    c = PresentationController()
    c.start()
    assert c.pause() is True
    assert c.pause() is False  # already paused
    assert c.stats.pauses == 1
    assert c.on_bot_stopped_speaking() is None  # nothing scheduled while paused
    assert c.resume() is None  # slide never started speaking: nothing to reschedule
    assert c.paused is False

    c.on_bot_started_speaking()
    c.on_bot_stopped_speaking()
    c.pause()
    action = c.resume()
    assert action is not None and action.kind is ActionKind.SCHEDULE_ADVANCE


def test_resume_while_bot_mid_utterance_does_not_schedule_advance():
    c = PresentationController()
    c.start()
    c.on_bot_started_speaking()
    c.pause()
    assert c.resume() is None
    assert c.bot_speaking is True


def test_pause_is_rejected_after_end():
    c = PresentationController()
    c.start()
    c.end()
    assert c.pause() is False


def test_stall_recovery():
    c = PresentationController()
    c.start()
    action = c.on_stall()
    assert action is not None and action.kind is ActionKind.PRESENT_SLIDE  # never started speaking
    c.on_bot_started_speaking()
    c.on_bot_stopped_speaking()
    action = c.on_stall()
    assert action is not None and action.kind is ActionKind.CONTINUE_SLIDE
    assert c.stats.stalls_recovered == 2
    c.pause()
    assert c.on_stall() is None


def test_qna_idle_prompts_then_ends_session():
    c = PresentationController(max_idle_prompts=2)
    run_full_deck(c)
    assert c.on_qna_idle().kind is ActionKind.IDLE_PROMPT
    assert c.on_qna_idle().kind is ActionKind.IDLE_PROMPT
    assert c.on_qna_idle().kind is ActionKind.END_SESSION


def test_user_activity_resets_idle_prompts():
    c = PresentationController(max_idle_prompts=1)
    run_full_deck(c)
    c.on_qna_idle()
    c.on_user_activity_in_qna()
    assert c.on_qna_idle().kind is ActionKind.IDLE_PROMPT


def test_idle_not_triggered_outside_qna():
    c = PresentationController()
    c.start()
    assert c.on_qna_idle() is None


def test_user_message_counting_ignores_blank():
    c = PresentationController()
    c.on_user_message("   ")
    c.on_user_message("what is a tsunami?")
    assert c.stats.user_turns == 1


def test_snapshot_shape():
    c = PresentationController()
    c.start()
    snap = c.snapshot()
    assert snap["mode"] == "presenting"
    assert snap["slide_number"] == 1
    assert snap["total_slides"] == len(DECK)
    assert snap["slide_title"] == DECK[0].title
    assert snap["talking_points"] == list(DECK[0].talking_points)
    assert snap["stats"]["slides_presented"] == 1
