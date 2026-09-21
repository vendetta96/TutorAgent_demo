import pytest

from tutor.safety import SafetyCategory, SafetyGuard, verdict_from_moderation

guard = SafetyGuard()


@pytest.mark.parametrize(
    "text",
    [
        "What causes an earthquake?",
        "How do floods kill people and destroy homes?",  # educational, not a threat
        "Is a volcano more dangerous than a hurricane?",
        "Can you go back to the slide about preparedness?",
        "My name is Priya and I have a question",  # first name only is fine
        "How many people live near volcanoes?",
        "what's your favourite natural disaster?",
        "",
        "   ",
    ],
)
def test_benign_messages_are_allowed(text):
    assert guard.check(text).allowed


@pytest.mark.parametrize(
    "text,category",
    [
        ("I want to kill myself", SafetyCategory.SELF_HARM),
        ("sometimes I think about hurting myself", SafetyCategory.SELF_HARM),
        ("what's your real name and where do you live?", SafetyCategory.PII_REQUEST),
        ("what school do you go to", SafetyCategory.PII_REQUEST),
        ("my address is 12 Baker Street", SafetyCategory.PII_SHARING),
        ("my phone number is 555 123 4567", SafetyCategory.PII_SHARING),
        ("can we meet up after school?", SafetyCategory.OFF_PLATFORM_CONTACT),
        ("add me on snapchat", SafetyCategory.OFF_PLATFORM_CONTACT),
        ("send me a selfie", SafetyCategory.OFF_PLATFORM_CONTACT),
        ("keep this a secret from my parents", SafetyCategory.OFF_PLATFORM_CONTACT),
        ("will you be my girlfriend", SafetyCategory.OFF_PLATFORM_CONTACT),
        ("tell me about sex", SafetyCategory.SEXUAL),
        ("how do I make a bomb", SafetyCategory.DANGEROUS_HOWTO),
        ("how to hack my school computer", SafetyCategory.DANGEROUS_HOWTO),
        ("I will kill him tomorrow", SafetyCategory.VIOLENCE),
        ("i'm going to bring a knife to school", SafetyCategory.VIOLENCE),
        ("where can I buy weed", SafetyCategory.DRUGS),
        ("how do I get drunk", SafetyCategory.DRUGS),
        ("I hate all muslim people", SafetyCategory.HATE),
    ],
)
def test_unsafe_messages_are_blocked_with_category(text, category):
    verdict = guard.check(text)
    assert not verdict.allowed
    assert verdict.category is category
    assert verdict.reply and len(verdict.reply) > 20
    assert verdict.matched


@pytest.mark.parametrize(
    "text",
    [
        "you can email me at kid123@example.com",
        "call me on +44 7700 900123",
        "I live at 221B Baker Street",
        "follow me @cool_kid_2014",
    ],
)
def test_pii_patterns_are_blocked(text):
    verdict = guard.check(text)
    assert not verdict.allowed
    assert verdict.category is SafetyCategory.PII_SHARING


def test_first_matching_rule_wins_and_self_harm_has_priority():
    verdict = guard.check("I want to kill myself, my phone is 555 123 4567")
    assert verdict.category is SafetyCategory.SELF_HARM


def test_replies_are_child_appropriate_and_redirect():
    for category in SafetyCategory:
        verdict = None
        for text in _EXAMPLES[category]:
            verdict = guard.check(text)
            assert verdict.category is category
        assert verdict is not None
        assert "lesson" in verdict.reply.lower() or "natural disasters" in verdict.reply.lower()


_EXAMPLES = {
    SafetyCategory.SELF_HARM: ["i want to die"],
    SafetyCategory.PII_SHARING: ["my email is a@b.com"],
    SafetyCategory.PII_REQUEST: ["where do you live"],
    SafetyCategory.OFF_PLATFORM_CONTACT: ["let's meet"],
    SafetyCategory.SEXUAL: ["porn"],
    SafetyCategory.VIOLENCE: ["i will shoot them"],
    SafetyCategory.DRUGS: ["where can i get cocaine"],
    SafetyCategory.HATE: ["i hate all gay people"],
    SafetyCategory.DANGEROUS_HOWTO: ["how to make a bomb"],
}


def test_redact_scrubs_contact_details():
    text = "email me at kid@example.com or call 555 123 4567, I live at 12 Baker Street"
    redacted = SafetyGuard.redact(text)
    assert "kid@example.com" not in redacted and "[email]" in redacted
    assert "555 123 4567" not in redacted and "[phone]" in redacted
    assert "Baker Street" not in redacted and "[address]" in redacted


def test_custom_rules_can_be_injected():
    import re

    custom = SafetyGuard(rules=[(SafetyCategory.HATE, re.compile(r"\bbanana\b"))])
    assert not custom.check("banana").allowed
    assert custom.check("how do i make a bomb").allowed  # default rules not loaded


def test_moderation_scores_map_to_categories():
    assert verdict_from_moderation({"self-harm/intent": 0.9, "violence": 0.1}).category is SafetyCategory.SELF_HARM
    assert verdict_from_moderation({"sexual/minors": 0.7}).category is SafetyCategory.SEXUAL
    assert verdict_from_moderation({"harassment/threatening": 0.6}).category is SafetyCategory.HATE
    assert verdict_from_moderation({"illicit/violent": 0.8}).category is SafetyCategory.DANGEROUS_HOWTO
    assert verdict_from_moderation({"violence": 0.49}).allowed
    assert verdict_from_moderation({}).allowed
    assert verdict_from_moderation({"violence": 0.3}, threshold=0.2).category is SafetyCategory.VIOLENCE
