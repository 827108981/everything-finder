"""仅供本机界面展示的一次性 RAG 候选快照。"""

from __future__ import annotations

from local_full_text_search.rag.models import EvidenceChunk


def candidate_snapshot(
    item: EvidenceChunk, *, selected: bool | None = None, reason: str = "",
    dedup_status: str = "", representative_chunk_id: str | None = None,
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
        "lexical_score_kind": item.lexical_score_kind,
        "lexical_contributions": list(item.lexical_contributions),
        "vector_rank": item.vector_rank,
        "vector_score": item.vector_score,
        "fused_rank": item.fused_rank,
        "lexical_route_rrf_score": item.lexical_route_rrf_score,
        "vector_route_rrf_score": item.vector_route_rrf_score,
        "lexical_route_hits": [
            {"route_id": hit.route_id, "rank": hit.rank, "score": hit.score,
             "score_kind": hit.score_kind} for hit in item.lexical_route_hits
        ],
        "vector_route_hits": [
            {"route_id": hit.route_id, "rank": hit.rank, "score": hit.score,
             "score_kind": hit.score_kind} for hit in item.vector_route_hits
        ],
        "vector_unit_hits": [
            {
                "unit_id": hit.unit_id,
                "route_id": hit.route_id,
                "raw_rank": hit.raw_rank,
                "score": hit.score,
                "kind": hit.kind,
                "start_char": hit.start_char,
                "end_char": hit.end_char,
                "token_count": hit.token_count,
                "text": hit.text,
            }
            for hit in item.vector_unit_hits
        ],
        "fused_score": item.fused_score,
        "rerank_score": item.rerank_score,
        "rerank_score_combined": item.rerank_score_combined,
        "answer_support_score": item.answer_support_score,
        "answer_support_reasons": list(item.answer_support_reasons),
        "constraint_score": item.constraint_score,
        "duplicate_of": item.duplicate_of,
        "source_count": len(item.all_sources),
        "member_chunk_ids": list(item.all_member_chunk_ids),
        "sources": [
            {"chunk_id": ref.stable_chunk_id, "filename": ref.filename,
             "location": ref.location_text}
            for ref in item.all_sources
        ],
        "dedup_status": dedup_status,
        "representative_chunk_id": (
            representative_chunk_id if dedup_status else item.chunk.stable_chunk_id
        ),
    }
    if item.constraint_decision is not None:
        decision = item.constraint_decision
        snapshot["constraint_decision"] = {
            "accepted": decision.accepted,
            "reason": decision.reason,
            "required_match_count": decision.required_match_count,
            "required_unknown_count": decision.required_unknown_count,
            "required_conflict_count": decision.required_conflict_count,
            "hard_match_count": decision.required_match_count,
            "hard_unknown_count": decision.required_unknown_count,
            "hard_conflict_count": decision.required_conflict_count,
            "soft_match_count": decision.soft_match_count,
            "soft_conflict_count": decision.soft_conflict_count,
            "hard_rejected": not decision.accepted,
            "support_status": decision.support_status,
            "support_spans": list(decision.support_spans),
            "judgements": [
                {
                    "kind": judgement.constraint.kind.value,
                    "value": judgement.constraint.value,
                    "polarity": judgement.constraint.polarity.value,
                    "strength": judgement.constraint.strength.value,
                    "source": judgement.constraint.source,
                    "source_text": judgement.constraint.source_text,
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
