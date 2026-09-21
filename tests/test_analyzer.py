
from tutor.improve.analyzer import analyze_directory, analyze_entries
from tutor.transcripts import TranscriptRecorder


def build_session(tmp_path, session_id="s1", uncertain=False):
    rec = TranscriptRecorder(tmp_path, session_id=session_id)
    rec.session_started(llm="gpt-4o")
    rec.event("action", kind="present_slide", reason="start", slide=1)
    rec.bot("Welcome everyone!", slide=1, mode="presenting")
    rec.event("action", kind="present_slide", reason="auto-advance", slide=2)
    rec.user("what is a tsunami?", slide=2, mode="presenting", interrupted=True)
    rec.bot(
        "I'm not sure about that exactly, but it is a big wave." if uncertain else "A tsunami is a giant wave. Back to our slide.",
        slide=2, mode="presenting", interrupted=False,
    )
    rec.event("pause", slide=2)
    rec.event("safety_block", category="pii_sharing", slide=2, mode="presenting")
    rec.event("action", kind="continue_slide", reason="stall: continue slide", slide=2)
    rec.event("action", kind="enter_qna", reason="last slide finished", slide=8)
    rec.user("ok thanks", slide=8, mode="qna")
    rec.bot("Any more questions?", slide=8, mode="qna")
    rec.user("can we go back to volcanoes", slide=8, mode="qna")
    rec.bot("Sure!", slide=8, mode="qna")
    rec.event("ending", inject_goodbye=True)
    rec.session_ended(
        metrics={"duration_seconds": 90.5, "estimated_cost_usd": 0.12, "llm_tokens": {"gpt-4o": {"prompt": 1000, "completion": 200}}},
        controller_stats={"pauses": 1},
    )
    return rec


def test_analyze_entries_extracts_pairs_and_counts(tmp_path):
    rec = build_session(tmp_path, uncertain=True)
    summary, pairs = analyze_entries(rec.entries, session_id="s1")
    assert summary.user_turns == 3 and summary.bot_turns == 4
    assert summary.interruptions == 1 and summary.pauses == 1 and summary.safety_blocks == 1
    assert summary.stalls == 1 and summary.slides_presented == 2
    assert summary.reached_qna and summary.ended_gracefully
    assert summary.duration_seconds == 90.5 and summary.estimated_cost_usd == 0.12
    assert summary.llm_prompt_tokens == 1000 and summary.llm_completion_tokens == 200
    assert [p.question for p in pairs] == ["what is a tsunami?", "ok thanks", "can we go back to volcanoes"]
    assert pairs[0].interrupted_bot and pairs[0].slide == 2 and pairs[0].uncertain
    assert not pairs[1].uncertain


def test_analyze_directory_groups_questions_by_slide(tmp_path):
    build_session(tmp_path, "a")
    build_session(tmp_path, "b", uncertain=True)
    analysis = analyze_directory(tmp_path)
    assert len(analysis.sessions) == 2
    assert len(analysis.qa_pairs) == 6
    assert len(analysis.uncertain_pairs) == 1
    # "ok thanks" is not a question; the other two are.
    assert analysis.questions_by_slide[2] == ["what is a tsunami?", "what is a tsunami?"]
    assert analysis.questions_by_slide[8] == ["can we go back to volcanoes", "can we go back to volcanoes"]
    assert analysis.safety_categories == {"pii_sharing": 2}
    d = analysis.as_dict()
    assert d["sessions"] == 2 and d["interruptions"] == 2 and d["sessions_reaching_qna"] == 2
    assert ("tsunami", 2) in d["top_question_words"]


def test_empty_directory(tmp_path):
    analysis = analyze_directory(tmp_path / "none")
    assert analysis.sessions == [] and analysis.as_dict()["qa_pairs"] == 0


def test_bot_turn_without_preceding_user_is_not_a_pair(tmp_path):
    rec = TranscriptRecorder(tmp_path, session_id="x")
    rec.bot("Hello", slide=1, mode="presenting")
    rec.bot("Still me", slide=1, mode="presenting")
    _, pairs = analyze_entries(rec.entries)
    assert pairs == []
