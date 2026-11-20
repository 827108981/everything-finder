import json

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.models import EvidenceChunk, RagChunk, RagQuery, SourceRef
from tools.source_bound_selection import parse_support, propose_selection


class Gateway:
    def __init__(self, *replies):
        self.replies = list(replies)

    def complete_text(self, request, token):
        return self.replies.pop(0)


def item(index, text, document=None):
    document = document or str(index)
    source = SourceRef(1, str(index), None, index, 1, f"/material/{document}.pptx",
                       f"{document}.pptx", ".pptx", (), f"第 {index} 张幻灯片")
    return EvidenceChunk(
        RagChunk(str(index), document, 0, text, text, f"hash-{index}", source),
        lexical_rank=index, rerank_score=0.8,
    )


def select(question, candidates, replies, *, per_doc=3):
    policy = EvidencePolicy(max_per_document=per_doc)
    query = RagQuery(question, final_top_k=8)
    baseline = policy.evaluate(query, candidates)
    return propose_selection(
        query, candidates, None, baseline=baseline, policy=policy,
        builder=ContextBuilder(), max_chars=12000,
        gateway=Gateway(*replies), model="fake", api_base_url="https://unused.invalid/v1",
        api_style="chat_completions", api_key="fake", timeout_seconds=5,
        token=CancelToken(),
    )


def test_quote_must_exist_in_its_own_candidate():
    candidates = [item(1, "低位开始", "same"), item(2, "空瓶切换", "same")]
    bad = json.dumps({"parts": [{"request_quote": "空瓶会怎样？",
                                "support": [{"candidate_id": "C2", "quote": "低位开始", "why": "错引"}]}]},
                     ensure_ascii=False)
    result = select("空瓶会怎样？", candidates, [bad], per_doc=1)
    assert result.reason.startswith("selector_fallback:")
    assert [x.chunk.stable_chunk_id for x in result.decision.accepted] == ["1"]


def test_extra_same_document_evidence_can_fill_an_empty_slot():
    candidates = [item(i, "无关内容", "same") for i in range(1, 5)]
    response = json.dumps({"parts": [{"request_quote": "空瓶会怎样？",
                                     "support": [{"candidate_id": "C4", "quote": "空瓶切换", "why": "直接支持"}]}]},
                          ensure_ascii=False)
    candidates[3].chunk = item(4, "空瓶切换", "same").chunk
    result = select("空瓶会怎样？", candidates, [response])
    assert result.reason == "proposed"
    assert {x.chunk.stable_chunk_id for x in result.decision.accepted} == {"1", "2", "3", "4"}


def test_unique_old_evidence_vetoes_replacement():
    candidates = [item(i, f"正文{i}") for i in range(1, 10)]
    candidates[-1].chunk = item(9, "新直接支持").chunk
    response = json.dumps({"parts": [{"request_quote": "新增事项？",
                                     "support": [{"candidate_id": "C9", "quote": "新直接支持", "why": "直接支持"}]}]},
                          ensure_ascii=False)
    audit = json.dumps({"safe": False, "reason": "旧段有独有事实", "unique_quote": "正文8"}, ensure_ascii=False)
    result = select("新增事项？", candidates, [response, audit])
    assert result.reason == "removal_not_safe_fallback"
    assert [x.chunk.stable_chunk_id for x in result.decision.accepted] == [str(i) for i in range(1, 9)]


def test_safe_replacement_retains_final_budget():
    candidates = [item(i, f"正文{i}") for i in range(1, 10)]
    candidates[-1].chunk = item(9, "新直接支持").chunk
    response = json.dumps({"parts": [{"request_quote": "新增事项？",
                                     "support": [{"candidate_id": "C9", "quote": "新直接支持", "why": "直接支持"}]}]},
                          ensure_ascii=False)
    audit = json.dumps({"safe": True, "reason": "旧段无关", "unique_quote": ""}, ensure_ascii=False)
    result = select("新增事项？", candidates, [response, audit])
    assert result.reason == "proposed"
    assert len(result.decision.accepted) == 8
    assert "9" in {x.chunk.stable_chunk_id for x in result.decision.accepted}
    assert "8" not in {x.chunk.stable_chunk_id for x in result.decision.accepted}


def test_invalid_removal_quote_falls_back():
    candidates = [item(i, f"正文{i}") for i in range(1, 10)]
    candidates[-1].chunk = item(9, "新直接支持").chunk
    response = json.dumps({"parts": [{"request_quote": "新增事项？",
                                     "support": [{"candidate_id": "C9", "quote": "新直接支持", "why": "直接支持"}]}]},
                          ensure_ascii=False)
    audit = json.dumps({"safe": False, "reason": "有独有事实", "unique_quote": "不在旧段里的假引文"}, ensure_ascii=False)
    result = select("新增事项？", candidates, [response, audit])
    assert result.reason.startswith("removal_audit_fallback:")
    assert "9" not in {x.chunk.stable_chunk_id for x in result.decision.accepted}
