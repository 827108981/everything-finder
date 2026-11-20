"""Conservative evidence structure; no domain dictionary or answer labels."""
from __future__ import annotations

import re
from local_full_text_search.rag.identifier_evidence import exact, structured_rows

_IDENTIFIER = re.compile(r'(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*(?![A-Za-z0-9_])')


def explicit_identifiers(question: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(m.group() for m in _IDENTIFIER.finditer(question)
                              if any(c.isdigit() for c in m.group())
                              or (m.group().isupper() and len(m.group()) >= 2)))


def evidence_features(question: str, text: str) -> dict:
    identifiers = explicit_identifiers(question)
    # Reuse the existing strict parser, independently of the optional domain adapter.
    # Only regular four-column labelled tables qualify; other layouts remain unknown.
    rows = [row for row in structured_rows(text)
            if any(exact(row['identifier'], value) for value in identifiers)]
    return {
        'identifiers': list(identifiers),
        'exact_matches': [value for value in identifiers if exact(text, value)],
        'verified_rows': rows,
        'structural_support': 'direct_labelled_record' if rows else 'unknown',
        'reason': 'Complete labelled record bound to a literal question identifier'
                  if rows else 'No unambiguous labelled record; no keyword reward',
    }
