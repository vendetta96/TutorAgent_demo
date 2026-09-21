import json

from tutor.transcripts import TranscriptRecorder, list_transcripts, load_transcript


def test_recorder_writes_jsonl_and_redacts_pii(tmp_path):
    rec = TranscriptRecorder(tmp_path, session_id="s1")
    rec.session_started(llm="gpt-4o")
    rec.user("my email is kid@example.com and what is lava?", slide=3, mode="presenting", interrupted=True)
    rec.bot("Lava is melted rock.", slide=3, mode="presenting")
    rec.event("pause", slide=3)
    rec.session_ended(metrics={"duration_seconds": 12.0}, controller_stats={"pauses": 1})

    assert rec.path == tmp_path / "s1.jsonl"
    lines = rec.path.read_text().strip().splitlines()
    assert len(lines) == 5
    entries = [json.loads(l) for l in lines]
    assert [e["type"] for e in entries] == ["session_start", "user", "bot", "event", "session_end"]
    assert entries[1]["text"] == "my email is [email] and what is lava?"
    assert entries[1]["interrupted_bot"] is True and entries[1]["slide"] == 3
    assert entries[2]["text"] == "Lava is melted rock."
    assert entries[3]["name"] == "pause"
    assert entries[4]["metrics"]["duration_seconds"] == 12.0
    assert all("ts" in e for e in entries)
    assert rec.entries == entries


def test_load_and_list(tmp_path):
    assert list_transcripts(tmp_path / "nope") == []
    a = TranscriptRecorder(tmp_path, session_id="a")
    a.session_started()
    b = TranscriptRecorder(tmp_path, session_id="b")
    b.session_started()
    (tmp_path / "notes.txt").write_text("ignored")
    assert [p.name for p in list_transcripts(tmp_path)] == ["a.jsonl", "b.jsonl"]
    assert load_transcript(a.path)[0]["session_id"] == "a"


def test_session_id_is_generated_when_missing(tmp_path):
    rec = TranscriptRecorder(tmp_path)
    assert rec.session_id and rec.path.suffix == ".jsonl"
    assert not rec.path.exists()  # nothing written until the first entry
