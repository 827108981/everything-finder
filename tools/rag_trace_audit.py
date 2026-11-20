"""Read-only checks linking original pages, AI index rows, and sent RAG text."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import posixpath
from xml.etree import ElementTree
from zipfile import ZipFile

from local_full_text_search.rag.content_identity import canonical_content
from local_full_text_search.rag.evaluation.evidence_matcher import map_case_evidence
from local_full_text_search.rag.evaluation.stage_scoring import score_stage

from tools.rag_eval_common import sha


_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


@lru_cache(maxsize=256)
def original_text(path: Path, page: int | None) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".pdf":
        import fitz

        if page is None:
            raise ValueError("PDF evidence requires a page number")
        with fitz.open(path) as document:
            return document[page - 1].get_text()
    if suffix == ".pptx":
        if page is None:
            raise ValueError("PPTX evidence requires a slide number")
        with ZipFile(path) as package:
            relationships = {
                item.attrib["Id"]: item.attrib["Target"]
                for item in ElementTree.fromstring(package.read("ppt/_rels/presentation.xml.rels"))
            }
            slide_ids = ElementTree.fromstring(package.read("ppt/presentation.xml")).findall(
                f".//{{{_P}}}sldId"
            )
            target = relationships[slide_ids[page - 1].attrib[f"{{{_R}}}id"]]
            name = target.lstrip("/") if target.startswith("/") else posixpath.normpath("ppt/" + target)
            slide = ElementTree.fromstring(package.read(name))
            body = "\n".join(
                "".join(node.text or "" for node in paragraph.findall(f".//{{{_A}}}t"))
                for paragraph in slide.findall(f".//{{{_A}}}p")
            ).strip()
            rels_name = posixpath.join(posixpath.dirname(name), "_rels", posixpath.basename(name) + ".rels")
            if rels_name in package.namelist():
                rels = ElementTree.fromstring(package.read(rels_name))
                for relation in rels:
                    if not relation.attrib.get("Type", "").endswith("/notesSlide"):
                        continue
                    note_target = relation.attrib["Target"]
                    note_name = (note_target.lstrip("/") if note_target.startswith("/") else
                                 posixpath.normpath(posixpath.join(posixpath.dirname(name), note_target)))
                    note = ElementTree.fromstring(package.read(note_name))
                    note_text = "\n".join(
                        "".join(node.text or "" for node in paragraph.findall(f".//{{{_A}}}t"))
                        for paragraph in note.findall(f".//{{{_A}}}p")
                    ).strip()
                    return "\n".join(part for part in (body, note_text) if part)
            return body
    if suffix == ".docx":
        with ZipFile(path) as package:
            root = ElementTree.fromstring(package.read("word/document.xml"))
        return "\n".join(
            "".join(node.text or "" for node in paragraph.iter(f"{{{_W}}}t"))
            for paragraph in root.iter(f"{{{_W}}}p")
        )
    raise ValueError(f"Unsupported original source: {path}")


def audit_case_sources(case, chunk_rows, actual_entries):
    """Verify exact source text; semantic applicability remains a separate review."""
    matches = map_case_evidence(case, chunk_rows)
    scored_entries = [
        {
            **entry,
            "member_chunk_ids": sorted(
                set(entry.get("member_chunk_ids", []))
                | {source["chunk_id"] for source in entry.get("sources", [])}
            ),
        }
        for entry in actual_entries
    ]
    completed = score_stage("context", scored_entries, matches)["gold_completion_ranks"]
    output = []
    for raw, match in zip(case.raw["gold_evidence"], (m for m in matches if m.kind == "positive")):
        context_rank = completed.get(match.evidence_id)
        confirmed_alternatives = [
            {
                "match_type": alternative.match_type,
                "chunk_ids": [part.chunk_id for part in getattr(alternative, "parts", ())],
            }
            for alternative in match.alternatives if alternative.confirmed
        ]
        path = Path(raw["file"])
        if not path.is_file():
            declared = {row["file_path"] for row in chunk_rows if Path(row["file_path"]).name == path.name}
            if len(declared) == 1:
                path = Path(next(iter(declared)))
        error = ""
        try:
            present = canonical_content(raw["quote"]) in canonical_content(
                original_text(path, int(raw["page"]) if raw.get("page") else None)
            )
        except Exception as exc:
            present, error = False, str(exc)[:300]
        output.append({
            "id": case.case_id,
            **raw,
            "resolved_path": str(path),
            "source_sha": sha(path) if path.is_file() else None,
            "original_quote": present,
            "original_error": error,
            "index_quote": any(alternative.confirmed for alternative in match.alternatives),
            "index_alternatives": confirmed_alternatives,
            "actual_context_quote": context_rank is not None,
            "actual_context_last_rank": context_rank,
            "actual_context_last_source_id": (
                scored_entries[context_rank - 1].get("source_id") if context_rank is not None else None
            ),
            "semantic_applicability": "needs_manual_review",
        })
    return output
