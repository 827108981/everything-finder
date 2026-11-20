"""Opt-in, source-traceable document encoding views. Never changes source text."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from typing import Callable


@dataclass(frozen=True, slots=True)
class VerifiedBackground:
    text: str
    source: str
    scope_authority: str = "local_explicit"


@dataclass(frozen=True, slots=True)
class EmbeddingContextPolicy:
    # These names describe isolated encoding contrasts, not domain-specific rules.
    view: str = "W0"
    background_tokens: int = 32

    def __post_init__(self):
        if self.view not in {f"W{i}" for i in range(6)}:
            raise ValueError("Unknown embedding view")
        if self.background_tokens < 1:
            raise ValueError("Background budget must be positive")

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps({"version": "encoding-context-v1", **asdict(self)}, sort_keys=True).encode()).hexdigest()

    def render(self, chunk, body: str, count_tokens: Callable[[str], int], max_tokens: int,
               background: VerifiedBackground | None = None) -> tuple[str, dict]:
        stem = Path(chunk.source.filename).stem.strip()
        location = chunk.source.location_text.strip()
        info = {"view": self.view, "scope_authority": "unknown", "context_source": "",
                "context_text": "", "context_omitted_reason": "", "body_unchanged": True}
        if self.view == "W4":
            return body, info
        if self.view == "W5":
            choices = ([background] if background and background.text.strip() else [])
            if stem:
                choices.append(VerifiedBackground(stem, "filename", "filename_hint"))
            reasons = []
            for field in choices:
                text = field.text.strip()
                if text in body:
                    reasons.append("exact_background_already_in_body")
                    # Do not replace an already present verified title with a broader filename.
                    break
                rendered = f"背景：{text}\n正文：{body}"
                if count_tokens(text) > self.background_tokens:
                    reasons.append("whole_background_exceeds_budget")
                    continue
                if count_tokens(rendered) > max_tokens:
                    reasons.append("whole_background_exceeds_total_budget")
                    continue
                info.update(scope_authority=field.scope_authority, context_source=field.source, context_text=text,
                            context_omitted_reason=";".join(reasons))
                return rendered, info
            info["context_omitted_reason"] = ";".join(reasons) or "no_verified_background_or_filename"
            return body, info
        fields = []
        if self.view in {"W0", "W2"} and stem:
            fields.append(f"来源：{stem}")
        if self.view in {"W0", "W1"} and location:
            fields.append(f"位置：{location}")
        rendered = "\n".join([*fields, f"正文：{body}"])
        if count_tokens(rendered) > max_tokens:
            info["context_omitted_reason"] = "legacy_total_budget_fallback"
            return body, info
        info.update(context_source="source_metadata", scope_authority="filename_hint" if stem and self.view in {"W0", "W2"} else "unknown",
                    context_text="\n".join(fields))
        return rendered, info


def fuse_parent_view_ranks(routes, *, top_k=50, rrf_k=60):
    """Equal-weight parent RRF; duplicate views and duplicate parents vote once.

    ``routes`` contains (view_fingerprint, ordered_parent_ids). View identity
    must describe the actual encoding, not an arbitrary route label.
    """
    seen_views=set();scores={};details={}
    for fingerprint, parents in routes:
        if fingerprint in seen_views:continue
        seen_views.add(fingerprint);seen_parents=set();rank=0
        for parent in parents:
            if parent in seen_parents:continue
            seen_parents.add(parent);rank+=1
            if rank>top_k:break
            scores[parent]=scores.get(parent,0.)+1./(rrf_k+rank)
            details.setdefault(parent,[]).append({"view":fingerprint,"rank":rank})
    ordered=sorted(scores,key=lambda parent:(-scores[parent],parent))[:top_k]
    return ordered,scores,details
