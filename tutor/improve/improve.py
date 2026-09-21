"""Learn from past transcripts (the open-ended goal).

    uv run python -m tutor.improve --dry-run        # deterministic analysis only, no API calls
    uv run python -m tutor.improve                  # judge answers, write FAQ + coaching notes
    uv run python -m tutor.improve --ingest         # ...and re-ingest the knowledge base

Outputs:
  knowledge/learned_faq.md            -> becomes retrievable knowledge (after ingest)
  data/learned/presenter_guidance.md  -> appended to the system prompt at startup
  data/learned/judgements.json        -> per-answer scores, for tracking over time
"""

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from tutor.config import Settings
from tutor.improve.analyzer import Analysis, QAPair, analyze_directory
from tutor.slides import DECK, deck_outline

JUDGE_INSTRUCTIONS = """You are reviewing a voice tutor that teaches 10-14 year olds about natural disasters.
For each question/answer pair, score the ANSWER from 1 (bad) to 5 (excellent) on:
- accuracy: factually correct
- age_appropriate: safe, simple, kind wording for children
- concise: short enough to be spoken (ideally under 4 sentences) while still answering
- on_topic: answers the question and, if the student interrupted a slide, returns to the lesson
Also write "better_answer": a short, spoken-style improved answer (2-4 sentences) if any score is below 4, else null.
Return JSON: {"judgements": [{"index": <int>, "accuracy": <int>, "age_appropriate": <int>, "concise": <int>, "on_topic": <int>, "better_answer": <string|null>, "note": <short string>}]}"""

FAQ_INSTRUCTIONS = """You write a study FAQ for a voice tutor about natural disasters for 10-14 year olds.
Given real student questions (with the slide they were asked on) and reviewer notes, produce Markdown:
- Group duplicate or similar questions together.
- One "## " heading per question (phrased clearly), followed by a 2-4 sentence correct, kid-friendly answer.
- Only include questions about natural disasters, science, geography, or safety. Skip chit-chat and unsafe topics.
- Use plain sentences, no lists inside answers. Facts must be accurate; if a question cannot be answered reliably, skip it.
Return only the Markdown, starting with "# Learned FAQ"."""

GUIDANCE_INSTRUCTIONS = """You coach a voice tutor (an LLM presenting slides about natural disasters to 10-14 year olds).
From the analysis and judgements below, write at most 8 short coaching bullets that will be appended to the tutor's
system prompt. Each bullet must be concrete and actionable, e.g. "On slide 4, students often ask what a tsunami is:
mention it briefly when covering types." or "Keep answers to interruptions under three sentences; several were too long."
Do not repeat the deck's existing rules; do not include anything unsafe or off-topic. Return Markdown bullets only."""


def _pairs_for_review(analysis: Analysis, limit: int) -> list[QAPair]:
    ordered = sorted(analysis.qa_pairs, key=lambda p: (not p.uncertain, not p.interrupted_bot))
    return ordered[:limit]


async def judge_pairs(client: AsyncOpenAI, model: str, pairs: list[QAPair], batch: int = 8) -> list[dict]:
    judgements: list[dict] = []
    for start in range(0, len(pairs), batch):
        chunk = pairs[start : start + batch]
        payload = [
            {
                "index": start + i,
                "slide": p.slide,
                "slide_title": DECK[p.slide - 1].title if 1 <= p.slide <= len(DECK) else None,
                "mode": p.mode,
                "student_interrupted_slide": p.interrupted_bot,
                "question": p.question,
                "answer": p.answer,
            }
            for i, p in enumerate(chunk)
        ]
        response = await client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_INSTRUCTIONS},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        data = json.loads(response.choices[0].message.content or "{}")
        for item in data.get("judgements", []):
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(pairs):
                item["question"] = pairs[idx].question
                item["answer"] = pairs[idx].answer
                item["slide"] = pairs[idx].slide
                judgements.append(item)
    return judgements


async def write_faq(client: AsyncOpenAI, model: str, analysis: Analysis, judgements: list[dict]) -> str:
    questions = [
        {
            "slide": slide,
            "slide_title": DECK[slide - 1].title if 1 <= slide <= len(DECK) else None,
            "questions": qs,
        }
        for slide, qs in sorted(analysis.questions_by_slide.items())
    ]
    notes = [
        {"question": j["question"], "better_answer": j.get("better_answer"), "note": j.get("note")}
        for j in judgements
        if j.get("better_answer")
    ]
    response = await client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": FAQ_INSTRUCTIONS},
            {
                "role": "user",
                "content": json.dumps({"deck": deck_outline(), "questions": questions, "reviewer_notes": notes}, ensure_ascii=False),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


async def write_guidance(client: AsyncOpenAI, model: str, analysis: Analysis, judgements: list[dict]) -> str:
    scores = {
        key: round(sum(j.get(key, 0) for j in judgements) / len(judgements), 2) if judgements else None
        for key in ("accuracy", "age_appropriate", "concise", "on_topic")
    }
    summary = {
        "analysis": analysis.as_dict(),
        "average_scores": scores,
        "low_scoring_examples": [
            {"slide": j["slide"], "question": j["question"], "note": j.get("note")}
            for j in judgements
            if min(j.get(k, 5) for k in ("accuracy", "age_appropriate", "concise", "on_topic")) <= 3
        ][:10],
        "deck": deck_outline(),
    }
    response = await client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": GUIDANCE_INSTRUCTIONS},
            {"role": "user", "content": json.dumps(summary, ensure_ascii=False)},
        ],
    )
    return (response.choices[0].message.content or "").strip()


async def run(args: argparse.Namespace) -> None:
    settings = Settings()
    analysis = analyze_directory(args.transcripts)
    print(json.dumps(analysis.as_dict(), indent=2))
    if not analysis.sessions:
        print("No transcripts found; run a few sessions first.")
        return
    if args.dry_run:
        for pair in analysis.uncertain_pairs[:10]:
            print(f"- uncertain on slide {pair.slide}: Q={pair.question!r} A={pair.answer[:120]!r}")
        return

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    pairs = _pairs_for_review(analysis, args.max_pairs)
    judgements = await judge_pairs(client, args.model, pairs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "judgements.json").write_text(json.dumps(judgements, indent=2, ensure_ascii=False))
    print(f"Judged {len(judgements)} answers -> {args.out_dir / 'judgements.json'}")

    if analysis.questions_by_slide:
        faq = await write_faq(client, args.model, analysis, judgements)
        faq_path = args.knowledge_dir / "learned_faq.md"
        faq_path.write_text(faq + "\n", encoding="utf-8")
        print(f"Wrote {faq_path} ({faq.count('## ')} entries)")

    guidance = await write_guidance(client, args.model, analysis, judgements)
    guidance_path = args.out_dir / "presenter_guidance.md"
    guidance_path.write_text(guidance + "\n", encoding="utf-8")
    print(f"Wrote {guidance_path}:\n{guidance}")

    if args.ingest:
        from tutor.knowledge.embedder import OpenAIEmbedder
        from tutor.knowledge.ingest import run_ingest

        await run_ingest(args.knowledge_dir, settings.knowledge_index, OpenAIEmbedder(api_key=settings.openai_api_key))


def main(argv: list[str] | None = None) -> None:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="Improve the tutor from past transcripts.")
    parser.add_argument("--transcripts", type=Path, default=Path("data/transcripts"))
    parser.add_argument("--knowledge-dir", type=Path, default=Path("knowledge"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/learned"))
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-pairs", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true", help="Analysis only; no OpenAI calls")
    parser.add_argument("--ingest", action="store_true", help="Re-ingest the knowledge base afterwards")
    args = parser.parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
