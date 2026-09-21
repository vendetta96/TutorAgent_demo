"""Deterministic guardrails for a classroom of underage students.

Rules run on the student's transcribed text before it reaches the LLM. A blocked
message never reaches the model; the agent speaks a fixed, safe reply instead.
This layer is intentionally rule-based so its behaviour is exact and testable;
the system prompt provides the softer, model-side layer on top.
"""

import re
from dataclasses import dataclass
from enum import Enum


class SafetyCategory(str, Enum):
    SELF_HARM = "self_harm"
    PII_SHARING = "pii_sharing"
    PII_REQUEST = "pii_request"
    OFF_PLATFORM_CONTACT = "off_platform_contact"
    SEXUAL = "sexual"
    VIOLENCE = "violence"
    DRUGS = "drugs"
    HATE = "hate"
    DANGEROUS_HOWTO = "dangerous_howto"


@dataclass(frozen=True)
class SafetyVerdict:
    allowed: bool
    category: SafetyCategory | None = None
    reply: str | None = None
    matched: str | None = None


_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s\-().]?){9,14}\d(?!\d)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_ADDRESS = re.compile(
    r"\b\d{1,5}\s+(?:[A-Za-z]+\s){0,3}(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|boulevard|blvd)\b\.?",
    re.IGNORECASE,
)
_SOCIAL_HANDLE = re.compile(r"(?:^|\s)@[A-Za-z0-9_.]{3,}")

_RULES: list[tuple[SafetyCategory, re.Pattern[str]]] = [
    (
        SafetyCategory.SELF_HARM,
        re.compile(
            r"\b(kill(ing)? myself|end(ing)? my life|hurt(ing)? myself|harm(ing)? myself|cut(ting)? myself|"
            r"want to die|suicid\w*|self[- ]harm\w*|don'?t want to (be alive|live))\b",
            re.IGNORECASE,
        ),
    ),
    (
        SafetyCategory.PII_REQUEST,
        re.compile(
            r"\b(what('?s| is) (your|ur) (real name|full name|address|phone|number|email|school|age|"
            r"birthday)|where do you live|what school do you go to)\b",
            re.IGNORECASE,
        ),
    ),
    (
        SafetyCategory.PII_SHARING,
        re.compile(
            r"\b(my (home |house )?address is|my phone( number)? is|my number is|my email is|"
            r"i live at|i go to \w+ (school|academy)|my (full|last) name is|my password is)\b",
            re.IGNORECASE,
        ),
    ),
    (
        SafetyCategory.OFF_PLATFORM_CONTACT,
        re.compile(
            r"\b(meet (me|up)( in person)?|let'?s meet|come to my house|add me on|"
            r"(instagram|snapchat|tiktok|discord|whatsapp|telegram|facebook)|send (me )?(a )?(photo|pic|picture|selfie)|"
            r"keep (this|it) (a )?secret|don'?t tell (your|my) (parents|teacher|mom|dad)|"
            r"(are you|be my) (girlfriend|boyfriend)|i love you)\b",
            re.IGNORECASE,
        ),
    ),
    (
        SafetyCategory.SEXUAL,
        re.compile(
            r"\b(sex|sexual|porn\w*|nude\w*|naked|horny|boobs|penis|vagina|masturbat\w*|hook ?up)\b",
            re.IGNORECASE,
        ),
    ),
    (
        SafetyCategory.DANGEROUS_HOWTO,
        re.compile(
            r"\b(how (do i|to|can i) (make|build|get|buy|find) (a |an )?(bomb|gun|weapon|explosive|knife|poison)|"
            r"how to (hack|hurt|kill|poison|stab|shoot)|make (a |an )?(bomb|explosive|poison))\b",
            re.IGNORECASE,
        ),
    ),
    (
        SafetyCategory.VIOLENCE,
        re.compile(
            r"\b(i('?ll| will| want to) (kill|shoot|stab|hurt|beat up) (you|him|her|them|someone|everyone)|"
            r"shoot up (the |my )?school|bring a (gun|knife) to school)\b",
            re.IGNORECASE,
        ),
    ),
    (
        SafetyCategory.DRUGS,
        re.compile(
            r"\b(where (can|do) i (get|buy) (weed|drugs|cocaine|alcohol|beer|vape\w*|cigarettes)|"
            r"how (to|do i) (get|use|smoke|take) (high|drunk|weed|drugs|cocaine|meth|vape\w*))\b",
            re.IGNORECASE,
        ),
    ),
    (
        SafetyCategory.HATE,
        re.compile(
            r"\b(i hate (all )?(black|white|asian|muslim|jewish|christian|hindu|gay|trans) people|"
            r"(\w+ )?(people|kids) are (stupid|subhuman|animals) because (they are|of their) (race|religion|colour|color))\b",
            re.IGNORECASE,
        ),
    ),
]

_REPLIES: dict[SafetyCategory, str] = {
    SafetyCategory.SELF_HARM: (
        "I'm really glad you told me, and I care about you. I'm just a lesson helper, so please talk to "
        "your teacher, a parent, or another trusted adult right now; they can help. "
        "Whenever you're ready, we can keep learning together."
    ),
    SafetyCategory.PII_REQUEST: (
        "I'm your lesson helper, so I don't have personal details to share, and you don't need to share "
        "yours either. Let's get back to our topic."
    ),
    SafetyCategory.PII_SHARING: (
        "Thanks, but you don't need to share personal information like that here, and it's safest not to. "
        "Let's keep going with the lesson."
    ),
    SafetyCategory.OFF_PLATFORM_CONTACT: (
        "I can only chat with you here during the lesson, and I can't do that. If anything is worrying you, "
        "talk to your teacher or a trusted adult. Now, back to natural disasters."
    ),
    SafetyCategory.SEXUAL: (
        "That's not something we talk about in this class. Let's stay with our lesson on natural disasters."
    ),
    SafetyCategory.DANGEROUS_HOWTO: (
        "I can't help with anything dangerous like that. Let's keep our focus on staying safe and on natural disasters."
    ),
    SafetyCategory.VIOLENCE: (
        "That sounds serious, and I can't help with hurting anyone. Please talk to a teacher or a trusted adult "
        "about how you're feeling. For now, let's return to our lesson."
    ),
    SafetyCategory.DRUGS: (
        "That's not something I can help with, and it's not safe. Let's get back to our lesson."
    ),
    SafetyCategory.HATE: (
        "In this class we treat everyone with respect, so I won't go along with that. Let's get back to "
        "learning about natural disasters."
    ),
}


class SafetyGuard:
    def __init__(self, rules: list[tuple[SafetyCategory, re.Pattern[str]]] | None = None):
        self._rules = rules if rules is not None else _RULES

    def check(self, text: str) -> SafetyVerdict:
        normalized = " ".join(text.split())
        if not normalized:
            return SafetyVerdict(allowed=True)

        for category, pattern in self._rules:
            match = pattern.search(normalized)
            if match:
                return SafetyVerdict(
                    allowed=False,
                    category=category,
                    reply=_REPLIES[category],
                    matched=match.group(0),
                )

        for pattern in (_PHONE, _EMAIL, _ADDRESS, _SOCIAL_HANDLE):
            match = pattern.search(normalized)
            if match:
                return SafetyVerdict(
                    allowed=False,
                    category=SafetyCategory.PII_SHARING,
                    reply=_REPLIES[SafetyCategory.PII_SHARING],
                    matched=match.group(0).strip(),
                )

        return SafetyVerdict(allowed=True)

    @staticmethod
    def redact(text: str) -> str:
        """Scrub obvious PII from text before it is stored in transcripts."""
        text = _EMAIL.sub("[email]", text)
        text = _PHONE.sub("[phone]", text)
        text = _ADDRESS.sub("[address]", text)
        return text


# Mapping from OpenAI moderation categories to our categories (checked in order).
_MODERATION_MAP: list[tuple[str, SafetyCategory]] = [
    ("self-harm", SafetyCategory.SELF_HARM),
    ("sexual", SafetyCategory.SEXUAL),
    ("violence", SafetyCategory.VIOLENCE),
    ("hate", SafetyCategory.HATE),
    ("harassment", SafetyCategory.HATE),
    ("illicit", SafetyCategory.DANGEROUS_HOWTO),
]


def verdict_from_moderation(category_scores: dict[str, float], threshold: float = 0.5) -> SafetyVerdict:
    """Turn an OpenAI moderation result into a verdict. Pure, so it is unit-testable."""
    for prefix, category in _MODERATION_MAP:
        for name, score in category_scores.items():
            if name.startswith(prefix) and (score or 0.0) >= threshold:
                return SafetyVerdict(
                    allowed=False, category=category, reply=_REPLIES[category], matched=f"moderation:{name}"
                )
    return SafetyVerdict(allowed=True)


class OpenAIModerator:
    """Optional second layer: OpenAI's moderation endpoint (enable with SAFETY_USE_MODERATION_API=true)."""

    def __init__(self, api_key: str | None = None, model: str = "omni-moderation-latest", threshold: float = 0.5):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._threshold = threshold

    async def check(self, text: str) -> SafetyVerdict:
        response = await self._client.moderations.create(model=self._model, input=text)
        result = response.results[0]
        scores = {k: float(v) for k, v in result.category_scores.model_dump().items() if v is not None}
        return verdict_from_moderation(scores, self._threshold)
