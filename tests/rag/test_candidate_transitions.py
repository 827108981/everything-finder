from __future__ import annotations

from local_full_text_search.rag.evaluation.transitions import build_candidate_transitions
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.models import EvidenceChunk, RagChunk, SourceRef


def test_records_rerank_cutoff_and_context_budget() -> None:
    stages = {
        "deduplicated": [{"chunk_id": "a"}, {"chunk_id": "b"}],
        "rerank": [{"chunk_id": "a"}],
        "constraints": [{"chunk_id": "a", "constraint_decision": {"accepted": True}}],
        "evidence": [{"chunk_id": "a", "selected": True, "final_rank": 1}],
        "context": [],
    }
    events = []
    rows = build_candidate_transitions(stages, events, reranker_top_k=1)
    assert next(item for item in rows if item["chunk_id"] == "b")["reason_code"] == "reranker_top_k_cutoff"
    assert next(item for item in rows if item["from_stage"] == "evidence")["reason_code"] == "context_budget_cutoff"


def test_merged_member_is_not_reported_as_lost_gold() -> None:
    stages = {
        "fusion": [{"chunk_id": "pdf"}, {"chunk_id": "pptx"}],
        "deduplicated": [{"chunk_id": "pdf", "member_chunk_ids": ["pdf", "pptx"]}],
    }
    rows = build_candidate_transitions(stages, [], reranker_top_k=20)
    merged = next(item for item in rows if item["chunk_id"] == "pptx")
    assert merged["status"] == "merged"
    assert merged["rank_after"] == 1


def test_context_builder_records_every_budget_excluded_candidate() -> None:
    evidence = []
    for index in range(3):
        source = SourceRef(
            source_generation=1, stable_chunk_id=f"c{index}", document_id=index,
            file_id=index, root_id=1, file_path=f"C:/资料/{index}.pdf",
            filename=f"{index}.pdf", extension=".pdf", block_ids=(index,),
            location_text="第 1 页",
        )
        evidence.append(EvidenceChunk(RagChunk(
            f"c{index}", f"doc{index}", index, "正文" * 200,
            "正文" * 200, f"hash{index}", source,
        )))
    context = ContextBuilder().build(evidence, max_chars=256)
    assert [row["chunk_id"] for row in context.entries] == ["c0"]
    assert [row["chunk_id"] for row in context.excluded_entries] == ["c1", "c2"]
    assert context.entries[0]["truncated"] is True
    stages = {
        "evidence": [{"chunk_id": f"c{index}", "selected": True} for index in range(3)],
        "context": [{"chunk_id": "c0", "truncated": True}],
    }
    transitions = build_candidate_transitions(
        stages, [{"context_excluded": list(context.excluded_entries)}],
        reranker_top_k=3,
    )
    by_id = {row["chunk_id"]: row for row in transitions}
    assert by_id["c0"]["reason_code"] == "context_text_truncated"
    assert all(by_id[key]["reason_code"] == "context_budget_cutoff" for key in ("c1", "c2"))
