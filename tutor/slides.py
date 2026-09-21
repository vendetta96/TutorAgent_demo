"""The presentation deck: ordered slides, each with a title and talking points."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Slide:
    number: int
    title: str
    talking_points: tuple[str, ...]
    keywords: tuple[str, ...] = field(default_factory=tuple)

    def as_prompt(self) -> str:
        points = "\n".join(f"- {p}" for p in self.talking_points)
        return f"SLIDE {self.number}: {self.title.upper()}\n{points}"


DECK: tuple[Slide, ...] = (
    Slide(
        1,
        "Welcome & Overview",
        (
            "Greet the class warmly and introduce today's topic: natural disasters.",
            "Explain that we will look at what natural disasters are, why they happen, "
            "their main types, how they affect people and the environment, and how we can stay safe.",
            "Tell the students they can ask questions at any time and you will always come back to the slides.",
        ),
        ("welcome", "overview", "introduction"),
    ),
    Slide(
        2,
        "What Are Natural Disasters",
        (
            "A natural disaster is an extreme natural event that causes major damage to life, "
            "property, or the environment.",
            "Examples: earthquakes, floods, hurricanes, volcanic eruptions, and droughts.",
            "They are caused by the Earth's own natural processes, not by people directly.",
        ),
        ("definition", "what is", "examples"),
    ),
    Slide(
        3,
        "Why Natural Disasters Happen",
        (
            "Movement of tectonic plates causes earthquakes and volcanic activity.",
            "Extreme weather patterns cause storms, floods, and droughts.",
            "Climate-related changes can make some events stronger or more frequent.",
            "Some disasters are sudden (earthquakes) while others develop slowly (droughts).",
        ),
        ("causes", "why", "tectonic", "weather", "climate"),
    ),
    Slide(
        4,
        "Major Types of Natural Disasters",
        (
            "The most common categories: earthquakes, floods, cyclones and hurricanes, wildfires, "
            "landslides, and volcanic eruptions.",
            "Each type has different causes and impacts depending on geography and climate.",
            "Give one quick, vivid example for two or three of the types.",
        ),
        ("types", "earthquake", "flood", "cyclone", "hurricane", "wildfire", "landslide", "volcano"),
    ),
    Slide(
        5,
        "Impact on People",
        (
            "Disasters can cause loss of life, injuries, and destroy homes.",
            "Families may be displaced and have to move to shelters.",
            "Healthcare, schools, and daily life get disrupted.",
            "Keep the tone caring and age-appropriate; avoid graphic detail.",
        ),
        ("impact", "people", "communities", "displacement"),
    ),
    Slide(
        6,
        "Environmental Effects",
        (
            "Wildfires destroy forests and animal habitats.",
            "Floods can drown habitats and pollute water sources.",
            "Landslides and floods cause soil erosion.",
            "Some disasters also reshape landscapes and create new ecosystems over time.",
        ),
        ("environment", "ecosystem", "habitat", "erosion", "pollution"),
    ),
    Slide(
        7,
        "Preparedness and Safety",
        (
            "Preparation reduces damage and saves lives.",
            "Early warning systems, evacuation plans, emergency kits, and community awareness all help.",
            "Practical tips: know your safe spot, keep an emergency kit, listen to official alerts.",
            "Education and planning are the key to disaster resilience.",
        ),
        ("preparedness", "safety", "warning", "evacuation", "emergency kit"),
    ),
    Slide(
        8,
        "Conclusion & Discussion",
        (
            "Natural disasters are powerful events with serious impacts on society and the environment.",
            "Preparedness, scientific understanding, and community cooperation make a big difference.",
            "Thank the class and invite them to ask questions or revisit any slide.",
        ),
        ("conclusion", "summary", "discussion"),
    ),
)


def get_slide(number: int) -> Slide:
    if number < 1 or number > len(DECK):
        raise IndexError(f"Slide {number} does not exist (deck has {len(DECK)} slides)")
    return DECK[number - 1]


def deck_outline() -> str:
    return "\n".join(f"{s.number}. {s.title}" for s in DECK)
