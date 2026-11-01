from __future__ import annotations

import math


SCORE_KIND = "lexical_bm25_fields_v2"


def inverse_document_frequency(document_count: int, document_frequency: int) -> float:
    total = max(1, int(document_count))
    frequency = min(total, max(0, int(document_frequency)))
    return math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))


def bm25_term_score(
    *, document_count: int, document_frequency: int, term_frequency: int,
    field_length: int, average_field_length: float, k1: float = 1.2, b: float = 0.75,
) -> float:
    tf = max(0, int(term_frequency))
    if tf == 0:
        return 0.0
    average = max(float(average_field_length), 1e-9)
    length = max(0, int(field_length))
    denominator = tf + float(k1) * (1.0 - float(b) + float(b) * length / average)
    if denominator <= 0:
        return 0.0
    return inverse_document_frequency(document_count, document_frequency) * tf * (float(k1) + 1.0) / denominator
