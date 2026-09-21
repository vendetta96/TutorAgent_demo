"""Text-only harness that runs the real prompts/tools through OpenAI and judges the replies.

Uses the exact system prompt, slide messages, presenter-state notes and tool
definitions the voice agent uses, minus audio. This keeps evals honest: a prompt
change that breaks behaviour shows up here before it shows up in a classroom.
"""

import json
import os
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from tutor.controller import PresentationController
from tutor.prompts import build_state_note, build_system_prompt, enter_qna_message, slide_message
from tutor.slides import DECK
from tutor.tools import build_tools

AGENT_MODEL = os.getenv("EVAL_AGENT_MODEL", os.getenv("OPENAI_LLM_MODEL", "gpt-4o"))
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")


class _NoopDriver:
    recorder = None

    def _cancel_advance(self):
        pass

    async def begin_ending(self, *, inject_goodbye: bool):
        pass


def openai_tools() -> list[dict]:
    controller = PresentationController()
    schema = build_tools(controller, _NoopDriver())  # type: ignore[arg-type]
    return [{"type": "function", "function": s.to_default_dict()} for s in schema.standard_tools]


@dataclass
class AgentReply:
    text: str
    tool_calls: list[dict] = field(default_factory=list)

    def called(self, name: str) -> dict | None:
        for call in self.tool_calls:
            if call["name"] == name:
                return call
        return None


class Scenario:
    """Builds a conversation the way the live pipeline would, then asks the agent to reply."""

    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client or AsyncOpenAI()
        self.controller = PresentationController()
        self.messages: list[dict] = [{"role": "system", "content": build_system_prompt()}]

    def start(self):
        self.controller.start()
        self.messages.append({"role": "system", "content": slide_message(DECK[0], first=True)})
        return self

    def present_through(self, last_slide: int, spoken: str | None = None):
        """Simulate slides 1..last_slide having been presented (each as one assistant turn)."""
        for n in range(1, last_slide + 1):
            if n == 1:
                self.start()
            else:
                self.controller.on_bot_started_speaking()
                self.controller.on_bot_stopped_speaking()
                action = self.controller.advance()
                assert action and action.slide
                self.messages.append({"role": "system", "content": slide_message(action.slide)})
            if n < last_slide or spoken is None:
                self.messages.append(
                    {"role": "assistant", "content": f"(Presented slide {n}: {DECK[n - 1].title}.)"}
                )
        if spoken is not None:
            self.controller.on_bot_started_speaking()
            self.messages.append({"role": "assistant", "content": spoken})
        return self

    def interrupt(self, user_text: str):
        """Student speaks while the bot is mid-slide."""
        self.controller.on_user_started_speaking()
        self.controller.on_user_stopped_speaking()
        self.messages.append({"role": "user", "content": user_text})
        return self

    def enter_qna(self):
        self.present_through(len(DECK))
        self.controller.on_bot_started_speaking()
        self.controller.on_bot_stopped_speaking()
        action = self.controller.advance()
        assert action and action.kind.value == "enter_qna"
        self.messages.append({"role": "system", "content": enter_qna_message()})
        self.messages.append(
            {"role": "assistant", "content": "And that's the end of our slides! Thank you all. Does anyone have a question?"}
        )
        return self

    def say(self, user_text: str):
        self.controller.on_user_message(user_text)
        self.messages.append({"role": "user", "content": user_text})
        return self

    async def reply(self, knowledge: list[str] | None = None) -> AgentReply:
        note = build_state_note(self.controller, knowledge)
        messages = self.messages + [{"role": "system", "content": note}]
        response = await self.client.chat.completions.create(
            model=AGENT_MODEL, messages=messages, tools=openai_tools(), temperature=0.3
        )
        choice = response.choices[0].message
        calls = [
            {"name": c.function.name, "arguments": json.loads(c.function.arguments or "{}")}
            for c in (choice.tool_calls or [])
        ]
        return AgentReply(text=(choice.content or "").strip(), tool_calls=calls)


JUDGE_SYSTEM = """You are a strict evaluator of a voice tutor for children aged 10-14 learning about natural disasters.
You will get the conversation context, the tutor's reply, and a list of criteria. For each criterion decide PASS or FAIL
and give a one-sentence reason. Judge only the reply. Return JSON:
{"results": [{"criterion": <string>, "pass": <bool>, "reason": <string>}]}"""


@dataclass
class Judgement:
    results: list[dict]

    @property
    def passed(self) -> bool:
        return all(r["pass"] for r in self.results)

    @property
    def failures(self) -> list[str]:
        return [f"{r['criterion']}: {r['reason']}" for r in self.results if not r["pass"]]


async def judge(client: AsyncOpenAI, context: str, reply: str, criteria: list[str]) -> Judgement:
    response = await client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": json.dumps({"context": context, "reply": reply, "criteria": criteria}, ensure_ascii=False),
            },
        ],
    )
    data = json.loads(response.choices[0].message.content or "{}")
    results = data.get("results", [])
    if len(results) != len(criteria):
        raise AssertionError(f"judge returned {len(results)} results for {len(criteria)} criteria: {data}")
    return Judgement(results)
