"""Prompt templates and verbalizers for ABSA prompt tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass
class PromptTemplate:
    """Format ABSA examples into textual prompts."""

    template: str = (
        "Review: {sentence}\n"
        "Aspect: {aspect}\n"
        "Sentiment (positive, neutral, negative):"
    )

    def render(self, sentence: str, aspect: str) -> str:
        return self.template.format(sentence=sentence, aspect=aspect)


@dataclass
class Verbalizer:
    """Map dataset labels to natural language tokens."""

    mapping: Dict[str, str]

    def __post_init__(self) -> None:
        if len(set(self.mapping.values())) != len(self.mapping):
            raise ValueError("Verbalizer tokens must be unique")

    def vocab(self) -> Iterable[str]:
        return self.mapping.values()

    def __call__(self, label: str) -> str:
        if label not in self.mapping:
            raise KeyError(f"Unknown label '{label}'")
        return self.mapping[label]


DEFAULT_TEMPLATE = PromptTemplate()
DEFAULT_VERBALIZER = Verbalizer(
    {
        "positive": "great",
        "neutral": "okay",
        "negative": "terrible",
    }
)
