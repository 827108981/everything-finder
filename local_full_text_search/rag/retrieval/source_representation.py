"""Conservative native source structure sidecars; no Gold or domain vocabulary."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

from .embedding_context import VerifiedBackground
from .vector_units import VectorRetrievalUnit

def normalized(text):
    return re.sub(r'\s+', '', text).casefold()


def inspect_pptx(path) -> dict:
    """Only real title placeholders and explicitly styled, simple native tables.

    Merged, blank/duplicate-header and unlabeled grids are retained in the audit
    but not guessed into header:value facts. Notes do not acquire slide titles.
    """
    from pptx import Presentation
    p = Path(path)
    result = {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "slides": {}}
    for number, slide in enumerate(Presentation(p).slides, 1):
        title = slide.shapes.title
        entry = {"title": title.text.strip() if title is not None else "",
                 "title_shape_id": title.shape_id if title is not None else None, "tables": []}
        # Grouped shapes have their own coordinate scopes. Do not infer a title
        # or header relation across those scopes in this initial capability.
        for shape in slide.shapes:
            if not shape.has_table: continue
            table = shape.table
            cells = [[c.text.strip() for c in row.cells] for row in table.rows]
            reason = ""
            if len(cells) < 2 or len(cells[0]) < 2: reason = "not_a_multirow_multicolumn_table"
            elif any(c.is_merge_origin or c.is_spanned for row in table.rows for c in row.cells): reason = "merged_cells_require_separate_layout_validation"
            elif not table.first_row: reason = "no_explicit_first_row_header_style"
            elif any(not h for h in cells[0]): reason = "blank_header"
            elif len({normalized(h) for h in cells[0]}) != len(cells[0]): reason = "duplicate_header"
            entry['tables'].append({"shape_id": shape.shape_id, "cells": cells,
                                    "eligible": not reason, "skip_reason": reason,
                                    "header_basis": "native_table_firstRow_and_nonblank_unique_cells"})
        result['slides'][str(number)] = entry
    return result


class SourceBackgroundProvider:
    def __init__(self, documents: dict):
        self.documents = documents
        self.fingerprint = hashlib.sha256(json.dumps(documents, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def __call__(self, chunk, start, end):
        source = chunk.source
        doc = self.documents.get(str(Path(source.file_path)))
        if not doc or not source.slide_number or '备注' in source.location_text or 'notes' in source.location_text.lower():
            return None
        slide = doc['slides'].get(str(source.slide_number))
        if not slide or not slide['title']: return None
        return VerifiedBackground(slide['title'], f"{doc['sha256']}:slide:{source.slide_number}:shape:{slide['title_shape_id']}")


class StructuredUnitBuilder:
    """Keep legacy windows and supplement complete, individually traceable rows."""
    def __init__(self, base, documents: dict, replacement=False):
        self.base = base
        self.documents = documents
        self.replacement = replacement
        self.fingerprint = hashlib.sha256((base.fingerprint + SourceBackgroundProvider(documents).fingerprint + str(replacement) + 'native-simple-row-v1').encode()).hexdigest()

    def build(self, chunk):
        windows = list(self.base.build(chunk))
        source = chunk.source
        doc = self.documents.get(str(Path(source.file_path)))
        if not doc or not source.slide_number or '备注' in source.location_text:
            return tuple(windows)
        slide = doc['slides'].get(str(source.slide_number))
        if not slide: return tuple(windows)
        seen = set()
        supplements = []
        for table in slide['tables']:
            if not table['eligible']: continue
            header = table['cells'][0]
            for r, values in enumerate(table['cells'][1:], 1):
                # Blank values are unknown, not zeros. Keep rows with at least
                # two actual values and require them in this original parent.
                nonblank = [v for v in values if v]
                if len(nonblank) < 2 or not all(normalized(v) in normalized(chunk.text) for v in nonblank): continue
                body = '\n'.join(f'{h}：{v}' for h, v in zip(header, values) if v)
                if body in seen: continue
                seen.add(body)
                if self.base.count_tokens(body) > self.base.policy.max_tokens: continue
                background = self.base.background_provider(chunk, 0, 0) if self.base.background_provider else None
                if self.base.context_policy:
                    encoded, meta = self.base.context_policy.render(chunk, body, self.base.count_tokens, self.base.policy.max_tokens, background)
                else:
                    encoded = self.base._embedding_text(chunk, body)
                    if self.base.count_tokens(encoded) > self.base.policy.max_tokens: encoded = body
                    meta = {}
                logical = hashlib.sha256(f"{doc['sha256']}:{source.slide_number}:{table['shape_id']}:{r}:{chunk.stable_chunk_id}".encode()).hexdigest()
                meta.update(logical_unit_id=logical, kind='native_table_row', source_sha256=doc['sha256'],
                            slide=source.slide_number, shape=table['shape_id'], row=r,
                            header_row=0, cells=[{"row": r, "column": c, "header": h, "value": v} for c,(h,v) in enumerate(zip(header,values))],
                            source_span_kind='noncontiguous_original_cells', body_unchanged=False)
                unit_id = hashlib.sha256((logical + self.fingerprint + encoded).encode()).hexdigest()
                supplements.append(VectorRetrievalUnit(unit_id, chunk.stable_chunk_id, len(windows)+len(supplements), body,
                    encoded, 'native_table_row', 0, 0, self.base.count_tokens(encoded), meta))
        if self.replacement and supplements:
            # Sensitivity only: same per-parent representation count as baseline.
            return tuple((supplements + windows)[:len(windows)])
        return tuple(windows + supplements)
