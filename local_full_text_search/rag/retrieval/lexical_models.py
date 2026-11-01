from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LexicalToken:
    text: str
    normalized: str
    start: int
    end: int
    surface_kind: str = "word"


@dataclass(frozen=True, slots=True)
class LexicalPhrase:
    normalized: str
    start: int
    end: int
    component_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LexicalQuery:
    original_text: str
    tokens: tuple[LexicalToken, ...]
    locator_phrases: tuple[LexicalPhrase, ...]
    duplicate_terms: tuple[str, ...]
    analyzer_fingerprint: str


@dataclass(frozen=True, slots=True)
class LexicalTermContribution:
    term: str
    document_frequency: int
    body_frequency: int
    title_frequency: int
    body_score: float
    title_score: float
    match_methods: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "term": self.term,
            "document_frequency": self.document_frequency,
            "body_frequency": self.body_frequency,
            "title_frequency": self.title_frequency,
            "body_score": self.body_score,
            "title_score": self.title_score,
            "match_methods": list(self.match_methods),
        }
