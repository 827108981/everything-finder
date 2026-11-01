from __future__ import annotations

from local_full_text_search.rag.evaluation.evidence_matcher import map_case_evidence
from local_full_text_search.rag.evaluation.metrics import score_case
from local_full_text_search.rag.evaluation.gold_loader import GoldCase


def _case(quote: str) -> GoldCase:
    return GoldCase(
        case_id="gold-1", group_id="pump", question="P02 是什么泵？",
        expected_behavior="answer", required_facts=(), forbidden_claims=(),
        constraints={}, missing_fields=(),
        gold_evidence=({"file": "机器.pdf", "page": 4, "quote": quote},),
        hard_negatives=(), corpus_snapshot="bundle", review_status="reviewed", raw={},
    )


def _row(chunk_id: str, text: str, filename: str, ordinal: int) -> dict:
    return {
        "chunk_id": chunk_id, "text": text, "filename": filename,
        "file_path": "E:/资料/" + filename, "page_number": 4,
        "location": "第 4 页", "ordinal": ordinal,
    }


def test_split_quote_requires_both_chunks_and_completion_rank() -> None:
    quote = "确认交换机已上电并检查端口灯"
    left = _row("left", "确认交换机已上电并检查", "机器.pdf", 0)
    right = _row("right", "端口灯", "机器.pdf", 1)
    mappings = map_case_evidence(_case(quote), [left, right])
    score = score_case(_case(quote), {"lexical": [right, left], "context": [{**left, "included_text": left["text"]}]}, mappings)
    assert score["stages"]["lexical"]["first_gold_rank"] == 2
    assert score["stages"]["context"]["gold_evidence_found"] == 0
    assert score["stages"]["context"]["first_gold_fragment_rank"] is not None


def test_split_quote_removes_verified_chunk_overlap() -> None:
    quote = "确认交换机已上电并检查端口灯状态"
    left = _row("left", "确认交换机已上电并检查端口", "机器.pdf", 0)
    right = _row("right", "已上电并检查端口灯状态", "机器.pdf", 1)
    mapping = map_case_evidence(_case(quote), [left, right])[0]
    assert len(mapping.alternatives) == 1
    assert [part.chunk_id for part in mapping.alternatives[0].parts] == ["left", "right"]
    assert mapping.alternatives[0].parts[1].source_start > 0


def test_evidence_part_keeps_original_offsets_for_line_broken_quote() -> None:
    quote = "清\n洗液泵"
    row = _row("pdf", "前文：清洗液泵负责排放。", "机器.pdf", 0)
    part = map_case_evidence(_case(quote), [row])[0].alternatives[0].parts[0]
    assert quote[part.quote_original_start:part.quote_original_end] == quote
    assert row["text"][part.source_original_start:part.source_original_end] == "清洗液泵"


def test_cross_format_full_body_equivalence_is_counted() -> None:
    quote = "P02废液泵负责排放废液"
    pdf = _row("pdf", quote, "机器.pdf", 0)
    pptx = _row("pptx", quote, "机器.pptx", 0)
    mappings = map_case_evidence(_case(quote), [pdf, pptx])
    assert mappings[0].equivalent_chunk_ids == ("pptx",)
    result = score_case(_case(quote), {"lexical": [pptx]}, mappings)
    assert result["stages"]["lexical"]["gold_evidence_found"] == 1
    assert result["stages"]["lexical"]["gold_match_types"]["gold-1:positive:1"] == "layout_equivalent"


def test_cross_format_one_percent_fact_change_stays_suspected() -> None:
    common = "废液泵执行排放流程" * 20
    quote = common + "流量为140"
    pdf = _row("pdf", quote, "机器.pdf", 0)
    pptx = _row("pptx", quote + "；其他流量为115", "机器.pptx", 0)
    mappings = map_case_evidence(_case(quote), [pdf, pptx])
    assert mappings[0].equivalent_chunk_ids == ()
    assert mappings[0].suspected_chunk_ids == ("pptx",)
    result = score_case(_case(quote), {"lexical": [pptx]}, mappings)
    assert result["stages"]["lexical"]["gold_evidence_found"] == 0
    assert result["stages"]["lexical"]["gold_evidence_total_declared"] == 1


def test_unmapped_gold_stays_in_denominator() -> None:
    case = _case("资料中尚不存在的引文")
    mapping = map_case_evidence(case, [])
    result = score_case(case, {"lexical": []}, mapping)
    assert result["stages"]["lexical"]["gold_evidence_recall"] == 0


def test_gold_page_number_does_not_match_suffix_of_another_page() -> None:
    case = _case("P02废液泵负责排放废液")
    row = _row("wrong-page", "P02废液泵负责排放废液", "机器.pdf", 0)
    row.update(page_number=19, location="第 19 页")
    from dataclasses import replace
    case = replace(case, gold_evidence=({"file": "机器.pdf", "page": 9,
                                         "quote": "P02废液泵负责排放废液"},))
    mapping = map_case_evidence(case, [row])[0]
    assert mapping.chunk_ids == ()
    assert mapping.confidence == "unmatched"
