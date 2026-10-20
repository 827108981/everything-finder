"""仅供本机界面展示的一次性 RAG 候选快照。"""

from __future__ import annotations

from local_full_text_search.rag.models import EvidenceChunk


def candidate_snapshot(
    item: EvidenceChunk, *, selected: bool | None = None, reason: str = ""
) -> dict[str, object]:
    source = item.chunk.source
    snapshot: dict[str, object] = {
        "chunk_id": item.chunk.stable_chunk_id,
        "document_key": item.chunk.document_key,
        "filename": source.filename,
        "file_path": source.file_path,
        "location": source.location_text,
        "page_number": source.page_number,
        "slide_number": source.slide_number,
        "sheet_name": source.sheet_name,
        "cell_start": source.cell_start,
        "cell_end": source.cell_end,
        "line_start": source.line_start,
        "line_end": source.line_end,
        "source_type": source.source_type,
        "text": item.chunk.text,
        "lexical_rank": item.lexical_rank,
        "lexical_score": item.lexical_score,
        "vector_rank": item.vector_rank,
        "vector_score": item.vector_score,
        "fused_score": item.fused_score,
        "rerank_score": item.rerank_score,
        "constraint_score": item.constraint_score,
        "duplicate_of": item.duplicate_of,
    }
    if item.constraint_decision is not None:
        decision = item.constraint_decision
        snapshot["constraint_decision"] = {
            "accepted": decision.accepted,
            "reason": decision.reason,
            "required_match_count": decision.required_match_count,
            "required_unknown_count": decision.required_unknown_count,
            "required_conflict_count": decision.required_conflict_count,
            "judgements": [
                {
                    "kind": judgement.constraint.kind.value,
                    "value": judgement.constraint.value,
                    "status": judgement.status.value,
                    "reason": judgement.reason,
                    "matched_text": judgement.matched_text,
                }
                for judgement in decision.judgements
            ],
        }
    if selected is not None:
        snapshot["selected"] = selected
    if reason:
        snapshot["reason"] = reason
    return snapshot
