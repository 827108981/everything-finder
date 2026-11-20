from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from local_full_text_search.rag.content_identity import canonical_content
from tools.rag_eval_common import lines, pair_order, read, redact, sha, write
from tools import rag_trace_audit


def test_common_io_matches_existing_experiment_format(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    value = {"question": "蠕动泵怎么检查？", "count": 2}
    write(target, value)
    assert target.read_text(encoding="utf-8") == json.dumps(
        value, ensure_ascii=False, indent=2
    ) + "\n"
    assert read(target) == value
    assert sha(target) == sha(str(target))

    rows = tmp_path / "runs.jsonl"
    rows.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")
    assert lines(rows, skip_blank=True) == [{"id": 1}, {"id": 2}]


def test_pair_order_and_secret_redaction() -> None:
    assert [pair_order(index, repeat) for index in range(2) for repeat in (1, 2, 3)] == [
        "AB", "BA", "AB", "BA", "AB", "BA"
    ]
    assert redact("failed: private-key", "private-key", limit=18) == "failed: [REDACTED]"


def test_trace_audit_does_not_count_page_only_context(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf placeholder")
    case = SimpleNamespace(
        case_id="case-1",
        raw={"gold_evidence": [{"file": str(source), "page": 1, "quote": "必要操作"}]},
    )
    match = SimpleNamespace(evidence_id="case-1:positive:1", kind="positive", alternatives=(
        SimpleNamespace(confirmed=True, match_type="declared", parts=()),
    ))
    monkeypatch.setattr(rag_trace_audit, "original_text", lambda path, page: "必要操作")
    monkeypatch.setattr(rag_trace_audit, "map_case_evidence", lambda case, rows: [match])

    def score(stage, entries, mappings):
        assert stage == "context"
        assert entries == [{"source_id": "S1", "included_text": "同页的其他内容", "member_chunk_ids": []}]
        return {"gold_completion_ranks": {}}

    monkeypatch.setattr(rag_trace_audit, "score_stage", score)
    result = rag_trace_audit.audit_case_sources(
        case, [], [{"source_id": "S1", "included_text": "同页的其他内容"}]
    )
    assert result[0]["original_quote"] is True
    assert result[0]["index_quote"] is True
    assert result[0]["actual_context_quote"] is False
    assert result[0]["actual_context_last_source_id"] is None
    assert result[0]["index_alternatives"] == [{"match_type": match.alternatives[0].match_type, "chunk_ids": []}]
    assert result[0]["semantic_applicability"] == "needs_manual_review"


def test_original_pptx_slide_includes_speaker_notes() -> None:
    source = Path("E:/生化资料/生化场景化培训教材-电解质V1.1.pptx")
    if not source.is_file():
        return
    text = rag_trace_audit.original_text(source, 20)
    assert canonical_content("先测试H-STD一次，以活化电极") in canonical_content(text)
