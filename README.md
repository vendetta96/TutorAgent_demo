# Tutor Bot

A voice AI teacher that presents an eight-slide lesson about natural disasters to a class of school students, answers their questions, and comes back to the slides. Built on **Pipecat** with **OpenAI** as the only external service.

## What it does

| Goal (from the assignment) | How it is met |
|---|---|
| Returns to topic after questions | A deterministic `PresentationController` tracks slide + interruption state; a `[PRESENTER STATE]` note is refreshed before every LLM call ("you were interrupted on slide 3, answer briefly, continue where you stopped"). Pipecat keeps only the spoken part of an interrupted reply in context, so "where you stopped" is exact. |
| Reliably ends the deck, enters Q&A, can jump back | One slide = one LLM turn. After the bot finishes a slide and 1.2 s of silence passes, the controller advances (no LLM reliance). After slide 8 it enters Q&A. LLM tools `go_to_slide`, `next_slide`, `end_session` handle "go back to the volcano slide" style requests. Idle prompts in Q&A, then a graceful goodbye. |
| Pause / resume exactly where it left off | Server-side `PauseGate` buffers every bot output frame between TTS and the transport; the browser suspends its `AudioContext`. Resume releases both. Mic is muted and interruptions suppressed while paused. |
| Guardrails for underage users | Deterministic `SafetyGuard` rules (self-harm, PII requests/sharing, off-platform contact, sexual, violence, drugs, hate, dangerous how-to). Blocked messages never reach the LLM; a fixed safe reply is spoken. A strict age-appropriate system prompt covers the rest. Transcripts are PII-redacted. |
| Deterministic tests | `pytest tests` — 115 tests, no network, ~5 s. Covers the controller, prompts, safety, RAG, metrics, transcripts, pause gate, driver timers and the real pipeline wiring. |
| LLM-as-a-judge evals | `pytest evals -m eval` — 10 scenarios run through the real prompts/tools via OpenAI, graded by a judge model. |
| Knowledge ingestion | `knowledge/*.md` → chunks → OpenAI embeddings → `data/knowledge_index.json` (numpy cosine search). Incremental: unchanged chunks are not re-embedded. Top-3 passages are injected into the prompt on every student question. |
| Metrics report on disconnect | TTFB and processing latency per service, user→bot response latency, tokens per model, TTS characters, STT seconds, tool calls, pauses, safety blocks, estimated cost. Printed on the console and shown in the UI. |
| **Open-ended: learn from old transcripts** | Every session is saved to `data/transcripts/*.jsonl`. `python -m tutor.improve` judges past answers, mines frequently asked questions into `knowledge/learned_faq.md` (re-ingested into RAG) and writes coaching bullets to `data/learned/presenter_guidance.md` (appended to the system prompt on the next run). |

## Setup

Requirements: Python 3.11 (uv installs it for you), Node 18+, an OpenAI API key.

```bash
# 1. Python side
uv sync                              # creates .venv with Python 3.11 and all deps
cp .env.example .env                 # then put your OPENAI_API_KEY in .env

# 2. Build the knowledge index (one-time, a fraction of a cent)
uv run python -m tutor.knowledge.ingest

# 3. Start the agent
uv run python main.py                # http://localhost:7860

# 4. Start the frontend (second terminal)
cd frontend
yarn install        # or: npm install
yarn dev            # or: npm run dev  -> http://localhost:5173
```

Open http://localhost:5173, press **Connect**, allow the microphone, and the lesson starts. Use headphones so the bot does not hear itself.

Model choice, pacing and thresholds are configurable in `.env` (see `.env.example`). `OPENAI_LLM_MODEL=gpt-4o-mini` is a cheap option for testing.

## Using it

- **Ask a question any time** — the bot answers and returns to the slide.
- **"Can you go back to the preparedness slide?"** — works during the presentation and in Q&A.
- **Pause / Resume** buttons — the bot freezes mid-sentence and continues from the same point.
- **After the last slide** the bot enters Q&A. Say "no more questions" to end the lesson; the metrics report prints in the server console and appears in the UI.

## Tests and evals

```bash
uv run pytest                        # deterministic tests (no API key needed)
uv run pytest evals -m eval -v       # LLM-as-a-judge evals (needs OPENAI_API_KEY, ~$0.05)
```

## Learning from transcripts

```bash
uv run python -m tutor.improve --dry-run     # analysis only, no API calls
uv run python -m tutor.improve --ingest      # judge answers, write FAQ + coaching notes, re-ingest
```

Then restart the agent: the coaching notes are appended to the system prompt and the FAQ is retrievable.

## Adding knowledge

Drop `.md` or `.txt` files into `knowledge/` and run `uv run python -m tutor.knowledge.ingest`. Try a query with `--query "what causes a tsunami"`.

## Project layout

```
main.py                     FastAPI server: /ws, /connect, /health, /report/latest
agent.py                    Pipeline wiring (the only Pipecat glue besides tutor/processors)
tutor/
  slides.py                 The deck (title + talking points per slide)
  controller.py             Pure state machine: modes, slide progression, pause, recovery
  prompts.py                System prompt, slide messages, per-turn presenter note
  safety.py                 Deterministic guardrails + PII redaction
  metrics.py                Metrics aggregation + console report
  transcripts.py            JSONL session recorder / loader
  tools.py                  LLM tools: go_to_slide, next_slide, end_session
  config.py                 Settings from environment
  knowledge/                chunking, vector store, OpenAI embedder, ingest CLI
  improve/                  transcript analyzer + LLM-powered improvement CLI
  processors/               Pipecat glue: PauseGate, SafetyProcessor, ContextAugmenter,
                            PresentationDriver (observer), MetricsObserver
knowledge/                  Markdown knowledge base
data/                       knowledge_index.json, transcripts/, learned/
tests/                      deterministic pytest suite
evals/                      LLM-as-a-judge evals
frontend/                   Vite + TypeScript client
```

## Pipeline

```
transport.in → OpenAI Realtime STT → SafetyProcessor → user aggregator (VAD, mute-while-paused)
  → ContextAugmenter (state note + RAG) → GPT-4o (tools) → OpenAI TTS → PauseGate
  → transport.out → assistant aggregator
observers: PresentationDriver (slide timing, watchdog, RTVI state), MetricsObserver
```

## Known limitations

- Slide progression is driven by the bot finishing its turn; a very long student silence mid-answer is treated as the end of the answer.
- Safety rules are regex-based (exact and testable) and will not catch every paraphrase; the system prompt is the second layer, and `SAFETY_USE_MODERATION_API=true` adds OpenAI's moderation endpoint as a third (adds ~200 ms per student turn).
- Pause relies on the browser suspending its AudioContext; a client that ignores that still stops within roughly the amount of audio it had buffered.
- The knowledge base is small and English-only.
