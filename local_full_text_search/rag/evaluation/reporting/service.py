from __future__ import annotations

import csv
import html
import json
import re
from dataclasses import dataclass, asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from local_full_text_search.rag.content_identity import canonical_content, content_identity
from local_full_text_search.rag.evaluation.gold_loader import GoldCase, GoldDataset, load_gold_dataset
from local_full_text_search.rag.evaluation.metrics import STAGE_LABELS, STAGE_ORDER

from .matching import EvidenceMatch, MatchConfig, highlight_segments, match_quote
from .navigation_assets import NAVIGATION_CSS, NAVIGATION_JS
from .overview_assets import OVERVIEW_JS
from .timing import case_timing, run_timing
from .timing_assets import TIMING_CSS, TIMING_JS


REPORT_SCHEMA_VERSION = 4
REPORT_MATCHER_VERSION = "evidence-equivalence-v2"
REPORT_METRIC_VERSION = "diagnostic-v2"


@dataclass(frozen=True, slots=True)
class ReportConfig:
    topk: int = 50
    default_visible: int = 10
    default_stage: str = "context"
    page_tolerance: int = 1
    high_match_threshold: float = 0.60
    partial_match_threshold: float = 0.30
    count_partial_matches: bool = False
    refusal_score_type: str = "rerank_score"
    refusal_score_threshold: float | None = None
    allow_snapshot_mismatch: bool = False

    def __post_init__(self) -> None:
        if int(self.topk) < 1:
            raise ValueError("报告 top-k 必须大于 0")
        if not 0 < int(self.default_visible) <= int(self.topk):
            raise ValueError("默认显示数量必须在 1 和 top-k 之间")
        if self.default_stage not in STAGE_ORDER:
            raise ValueError(f"未知报告阶段：{self.default_stage}")
        if self.refusal_score_type not in {"vector_score", "fused_score", "rerank_score"}:
            raise ValueError("拒答风险分数类型不受支持")

    @property
    def matcher(self) -> MatchConfig:
        return MatchConfig(
            page_tolerance=self.page_tolerance,
            high_threshold=self.high_match_threshold,
            partial_threshold=self.partial_match_threshold,
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_results(results_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(results_path).expanduser().resolve()
    if path.is_dir():
        run_path = path / "run.json"
        cases_path = path / "cases.jsonl"
    else:
        cases_path = path
        run_path = path.with_name("run.json")
    if not cases_path.is_file():
        raise FileNotFoundError(f"找不到逐题评测结果：{cases_path}")
    metadata = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
    results: list[dict[str, Any]] = []
    for line_number, line in enumerate(cases_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"逐题结果第 {line_number} 行不是有效 JSON：{exc.msg}") from exc
        if isinstance(row, dict):
            results.append(row)
    return metadata, results


def _bundle_from_path(value: Any) -> str:
    match = re.search(r"(g\d{6}-[a-z0-9]+)", str(value or ""), re.IGNORECASE)
    return match.group(1) if match else ""


def _snapshot_validation(dataset: GoldDataset, metadata: dict[str, Any], results: list[dict[str, Any]], config: ReportConfig) -> dict[str, Any]:
    declared = _bundle_from_path((dataset.metadata.get("_阅读说明") or {}).get("AI索引目录"))
    case_snapshots = sorted({str(case.corpus_snapshot) for case in dataset.cases if case.corpus_snapshot})
    run_bundle = str(metadata.get("ai_bundle_id") or "")
    event_bundles = sorted({
        str(event.get("rag_bundle_id") or "")
        for result in results
        for event in result.get("events", [])
        if isinstance(event, dict) and event.get("rag_bundle_id")
    })
    observed = sorted({item for item in [run_bundle, *event_bundles] if item})
    expected = sorted({item for item in [declared, *case_snapshots] if item})
    missing = not bool(observed) or not bool(expected)
    mismatch = missing or set(expected) != set(observed)
    warning = ""
    if not observed:
        warning = "逐题结果没有记录实际 AI 索引 bundle_id，无法完成运行版本校验。"
    elif not expected:
        warning = "Gold/阅读说明没有声明 corpus_snapshot 或 AI 索引 bundle_id，无法完成运行版本校验。"
    elif mismatch:
        warning = f"Gold/阅读说明索引为 {', '.join(expected)}，本次结果使用 {', '.join(observed)}。"
    return {
        "declared_bundle_id": declared,
        "case_snapshot_ids": case_snapshots,
        "run_bundle_id": run_bundle,
        "event_bundle_ids": event_bundles,
        "observed_bundle_ids": observed,
        "mismatch": mismatch,
        "missing": missing,
        "allowed": bool(config.allow_snapshot_mismatch),
        "ok": not mismatch and not missing,
        "warning": warning,
    }


def _candidate_rank(stage: str, candidate: dict[str, Any], index: int) -> int | None:
    if stage == "lexical" and candidate.get("lexical_rank") is not None:
        return int(candidate["lexical_rank"])
    if stage == "vector" and candidate.get("vector_rank") is not None:
        return int(candidate["vector_rank"])
    if stage == "evidence":
        if candidate.get("selected") is True and candidate.get("final_rank") is not None:
            return int(candidate["final_rank"])
        return None
    if stage == "constraints" and candidate.get("selected") is False:
        return None
    return index + 1


def _stage_executed(result: dict[str, Any], stage: str) -> bool:
    if stage in result and isinstance(result.get(stage), dict):
        return bool(result[stage].get("executed") or result[stage].get("candidates"))
    if stage in result.get("candidate_stages", {}):
        return True
    if stage == "context":
        return bool(result.get("answer", {}).get("sources")) or any(
            str(event.get("stage") or "") == "retrieval_completed"
            for event in result.get("events", [])
            if isinstance(event, dict)
        )
    return False


def _evidence_entries(case: GoldCase, kind: str) -> list[dict[str, Any]]:
    source = case.gold_evidence if kind == "positive" else case.hard_negatives
    return [
        {
            "evidence_id": f"{case.case_id}:{kind}:{index}",
            "kind": kind,
            **{str(key): _json_safe(value) for key, value in item.items()},
        }
        for index, item in enumerate(source, 1)
    ]


def _match_stage_candidates(
    case: GoldCase,
    stage: str,
    candidates: list[dict[str, Any]],
    config: ReportConfig,
    formal_mappings: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[EvidenceMatch]]]:
    positives = _evidence_entries(case, "positive")
    negatives = _evidence_entries(case, "hard_negative")
    all_evidence = positives + negatives
    matches_by_evidence: dict[str, list[EvidenceMatch]] = {item["evidence_id"]: [] for item in all_evidence}
    normalized_candidates: list[dict[str, Any]] = []
    for index, original in enumerate(candidates[: config.topk]):
        candidate = dict(original)
        if stage == "context" and candidate.get("included_text") is not None:
            # 上下文预算可能只纳入正文前缀；匹配和高亮必须以实际入模文字为准。
            candidate["full_text"] = candidate.get("text")
            candidate["text"] = candidate["included_text"]
        candidate["stage_rank"] = _candidate_rank(stage, candidate, index)
        identity_text = str(candidate.get("full_text") or candidate.get("text") or "")
        canonical = canonical_content(identity_text)
        candidate["content_group_id"] = (
            content_identity(identity_text)
            if len(canonical) >= 24 else str(candidate.get("chunk_id") or "")
        )
        candidate["rank_display"] = candidate["stage_rank"] if candidate["stage_rank"] is not None else "未入选"
        candidate_matches: list[EvidenceMatch] = []
        for evidence in all_evidence:
            match = match_quote(
                candidate,
                evidence,
                kind=str(evidence["kind"]),
                evidence_id=str(evidence["evidence_id"]),
                config=config.matcher,
            )
            matches_by_evidence[match.evidence_id].append(match)
            if match.status in {"exact", "high", "partial"}:
                candidate_matches.append(match)
        matched = [item for item in candidate_matches if item.status in {"exact", "high", "partial"}]
        countable = [item for item in matched if item.is_countable or (config.count_partial_matches and item.status == "partial")]
        gold = [item for item in countable if item.kind == "positive"]
        negatives_found = [item for item in countable if item.kind == "hard_negative"]
        matched_gold = [item for item in matched if item.kind == "positive"]
        matched_negatives = [item for item in matched if item.kind == "hard_negative"]
        if matched_gold and matched_negatives:
            category = "mixed"
        elif gold:
            category = "gold"
        elif negatives_found:
            category = "negative"
        elif matched_gold:
            category = "partial_gold"
        elif matched_negatives:
            category = "partial_negative"
        else:
            category = "unlabelled"
        candidate["category"] = category
        candidate["matches"] = [item.to_dict() for item in candidate_matches]
        if formal_mappings:
            member_ids = {str(candidate.get("chunk_id") or "")}
            member_ids.update(str(value) for value in candidate.get("member_chunk_ids", []))
            declared = [item for item in formal_mappings if item.get("kind") == "positive"
                        and member_ids.intersection(map(str, item.get("declared_chunk_ids", [])))]
            equivalent = [item for item in formal_mappings if item.get("kind") == "positive"
                          and member_ids.intersection(map(str, item.get("equivalent_chunk_ids", [])))]
            suspected = [item for item in formal_mappings if item.get("kind") == "positive"
                         and member_ids.intersection(map(str, item.get("suspected_chunk_ids", [])))]
            formal_positive = declared or equivalent
            candidate["formal_gold_type"] = (
                "指定金标" if declared else "排版等价金标" if equivalent else
                "疑似金标（不计分）" if suspected else ""
            )
            if formal_positive:
                full = any(
                    canonical_content(str(item.get("quote") or ""))
                    in canonical_content(str(candidate.get("text") or ""))
                    for item in formal_positive
                )
                candidate["category"] = "mixed" if matched_negatives else "gold" if full else "partial_gold"
            elif suspected:
                candidate["category"] = "partial_gold"
            elif category in {"gold", "partial_gold", "mixed"}:
                candidate["category"] = "negative" if matched_negatives else "unlabelled"
            confirmed_ids = {str(item.get("evidence_id") or "") for item in formal_positive}
            for match in candidate["matches"]:
                if match.get("kind") == "positive" and match.get("evidence_id") not in confirmed_ids:
                    match["status"] = "suspected" if suspected else "unmatched"
        highlight_matches = candidate_matches
        if formal_mappings:
            confirmed_ids = {
                str(item.get("evidence_id") or "") for item in formal_mappings
                if item.get("kind") == "positive" and (
                    member_ids.intersection(map(str, item.get("declared_chunk_ids", [])))
                    or member_ids.intersection(map(str, item.get("equivalent_chunk_ids", [])))
                )
            }
            suspected_ids = {
                str(item.get("evidence_id") or "") for item in formal_mappings
                if item.get("kind") == "positive"
                and member_ids.intersection(map(str, item.get("suspected_chunk_ids", [])))
            }
            highlight_matches = [
                replace(match, status="partial")
                if match.kind == "positive" and match.evidence_id in suspected_ids
                else match
                for match in candidate_matches
                if match.kind != "positive" or match.evidence_id in confirmed_ids | suspected_ids
            ]
        candidate["highlight_segments"] = highlight_segments(
            str(candidate.get("text") or ""), highlight_matches
        )
        candidate["status"] = _candidate_status(stage, candidate)
        normalized_candidates.append(candidate)
    return normalized_candidates, matches_by_evidence


def _candidate_status(stage: str, candidate: dict[str, Any]) -> str:
    if stage == "evidence" and candidate.get("selected") is not True:
        return "not_selected"
    if stage == "context":
        return "in_context"
    decision = candidate.get("constraint_decision") or {}
    if stage == "constraints" and decision.get("accepted") is False:
        return "rejected"
    if candidate.get("duplicate_of"):
        return "merged"
    return "present"


def _rank_map(matches: list[EvidenceMatch], candidates: list[dict[str, Any]], config: ReportConfig) -> dict[str, list[int]]:
    lookup = {str(item.get("chunk_id") or ""): item for item in candidates}
    result: dict[str, list[int]] = {}
    for match in matches:
        if not match.is_countable and not (config.count_partial_matches and match.status == "partial"):
            continue
        # match rows are attached to candidate details below; this function is kept for clarity.
        _ = lookup
    return result


def _case_diagnostics(
    case: GoldCase,
    stage_data: dict[str, dict[str, Any]],
    config: ReportConfig,
    official_stages: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    positives = [item for item in case.gold_evidence]
    positive_count = len(positives)

    def ranks(kind: str, stage: str) -> dict[str, list[int]]:
        official = (official_stages or {}).get(stage, {})
        if kind == "positive" and "gold_completion_ranks" in official:
            return {
                str(evidence_id): [int(rank)]
                for evidence_id, rank in official["gold_completion_ranks"].items()
            }
        output: dict[str, list[int]] = {}
        for candidate in stage_data.get(stage, {}).get("candidates", []):
            for item in candidate.get("matches", []):
                if item.get("kind") != kind:
                    continue
                if item.get("status") not in {"exact", "high"} and not (config.count_partial_matches and item.get("status") == "partial"):
                    continue
                rank = candidate.get("stage_rank")
                if rank is not None:
                    output.setdefault(str(item.get("evidence_id")), []).append(int(rank))
        return output

    def first_positive(stage: str) -> int | None:
        values = [rank for ranks_for_evidence in ranks("positive", stage).values() for rank in ranks_for_evidence]
        return min(values) if values else None

    target = config.default_stage
    target_ranks = ranks("positive", target)
    target_found = len(target_ranks)
    first = first_positive(target)
    upstream_found = any(
        ranks("positive", stage) for stage in ("lexical", "vector", "fusion")
    )
    suspected_candidates = [
        candidate for stage in ("lexical", "vector", "fusion")
        for candidate in stage_data.get(stage, {}).get("candidates", [])
        if candidate.get("formal_gold_type") == "疑似金标（不计分）"
    ]
    if positive_count and target_found == 0 and (_stage_executed(stage_data, target) or upstream_found):
        if suspected_candidates and not upstream_found:
            diagnostics.append(_diag(
                "suspected_gold", "疑似金标已召回，待核实",
                "同名跨格式资料包含完整引文，但整块正文存在差异；已展示名次，正式指标暂不计分。",
                case, "lexical", target, "high",
                (str(item.get("chunk_id") or "") for item in suspected_candidates),
            ))
        elif upstream_found:
            diagnostics.append(_diag(
                "gold_lost_after_recall", "金标在后续阶段退出",
                "金标曾进入关键词、向量或融合候选；请查看逐条退出原因。",
                case, "fusion", target, "high", (),
            ))
        else:
            diagnostics.append(_diag("not_recalled", "两路未召回", "金标没有进入已记录的关键词或向量深度。", case, "lexical", "vector", "high", ()))
    elif positive_count and target_found < positive_count:
        diagnostics.append(_diag("partial_coverage", "部分覆盖", f"目标阶段只命中 {target_found}/{positive_count} 条金标证据。", case, target, target, "high", tuple(target_ranks)))
    if first is not None and 6 <= first <= config.topk:
        diagnostics.append(_diag("late_gold", "召回但排后", f"目标阶段首条金标排在第 {first} 位。", case, target, target, "medium", tuple(target_ranks)))

    hard = ranks("hard_negative", target)
    hard_first = min((rank for values in hard.values() for rank in values), default=None)
    if hard_first is not None and (first is None or hard_first < first):
        diagnostics.append(_diag("negative_before_gold", "负例抢位", f"指定负例首位第 {hard_first}，早于金标第 {first or '未命中'}。", case, target, target, "high", tuple(hard)))

    mixed_ids = tuple(
        str(candidate.get("chunk_id") or "")
        for payload in stage_data.values()
        for candidate in payload.get("candidates", [])
        if candidate.get("category") == "mixed"
    )
    mixed_ids = tuple(dict.fromkeys(item for item in mixed_ids if item))
    if mixed_ids:
        diagnostics.append(_diag("mixed_chunk", "同页难区分", "同一内容块同时匹配金标和指定负例，需要查看块内高亮。", case, target, target, "high", mixed_ids))

    upper = stage_data.get("constraints", {}).get("candidates", [])
    final_ids = {str(item.get("chunk_id") or "") for item in stage_data.get("evidence", {}).get("candidates", []) if item.get("selected") is True}
    upstream_gold = {
        str(candidate.get("chunk_id") or "")
        for candidate in upper
        if candidate.get("category") in {"gold", "mixed"}
    }
    lost = tuple(sorted(item for item in upstream_gold if item not in final_ids))
    if lost:
        diagnostics.append(_diag("evidence_loss", "最终筛选丢失证据", f"条件检查已有金标片段，最终证据少了 {len(lost)} 条。", case, "constraints", "evidence", "high", lost))

    context_ids = {str(item.get("chunk_id") or "") for item in stage_data.get("context", {}).get("candidates", [])}
    final_gold = {
        str(candidate.get("chunk_id") or "")
        for candidate in stage_data.get("evidence", {}).get("candidates", [])
        if candidate.get("selected") is True and candidate.get("category") in {"gold", "mixed"}
    }
    if final_gold - context_ids:
        diagnostics.append(_diag("context_loss", "上下文未纳入", "最终证据中有金标片段没有进入实际上下文。", case, "evidence", "context", "high", tuple(final_gold - context_ids)))

    if case.expected_behavior == "refuse":
        accepted_context = stage_data.get("context", {}).get("candidates", [])
        if accepted_context:
            diagnostics.append(_diag(
                "refusal_context", "拒答题仍有可用上下文",
                f"检索模式下有 {len(accepted_context)} 条代表正文进入最终上下文；需核查设备及事实适用性门禁。",
                case, "evidence", "context", "high",
                (str(item.get("chunk_id") or "") for item in accepted_context),
            ))
        refusal_candidates = stage_data.get("context", {}).get("candidates", [])[:5]
        refusal_negatives = [item for item in refusal_candidates if item.get("category") in {"negative", "mixed"}]
        score_hits = []
        if config.refusal_score_threshold is not None:
            for item in stage_data.get("rerank", {}).get("candidates", []):
                score = item.get(config.refusal_score_type)
                if score is not None and float(score) >= config.refusal_score_threshold:
                    score_hits.append(str(item.get("chunk_id") or ""))
        if refusal_negatives or score_hits:
            diagnostics.append(_diag("refusal_risk", "拒答题风险", "拒答题的相似负例进入了前五，或达到配置的风险分数阈值。", case, "context", "context", "high", tuple(str(item.get("chunk_id") or "") for item in refusal_negatives + stage_data.get("rerank", {}).get("candidates", []) if str(item.get("chunk_id") or "") in score_hits)))

    if not diagnostics:
        if case.expected_behavior == "refuse":
            diagnostics.append(_diag(
                "no_defined_risk", "拒答题无可用上下文",
                "相似候选可供诊断，但没有代表正文进入最终上下文；检索模式不核查最终拒答措辞。",
                case, "constraints", "context", "low", (),
            ))
        else:
            diagnostics.append(_diag("no_defined_risk", "正常", "当前规则没有发现已定义的检索风险，不能等同于答案正确。", case, None, None, "low", ()))
    return diagnostics


def _diag(code: str, title: str, observed: str, case: GoldCase, stage_from: str | None, stage_to: str | None, priority: str, ids: Iterable[str]) -> dict[str, Any]:
    targets = {
        "not_recalled": ("QueryInterpreter", "LexicalRetriever", "VectorRetriever", "ChunkBuilder"),
        "partial_coverage": ("两路召回与证据覆盖映射",),
        "late_gold": ("重排输入和排在金标前的候选",),
        "negative_before_gold": ("负例候选的文件、页码和业务条件",),
        "mixed_chunk": ("内容块边界、表格行和业务条件",),
        "evidence_loss": ("EvidencePolicy.evaluate", "最终 TopK 和文档数量限制"),
        "context_loss": ("ContextBuilder.build", "上下文字符预算和去重"),
        "refusal_risk": ("拒答证据门禁和跨业务范围条件",),
        "refusal_context": ("QueryInterpreter 设备适用性", "ConstraintMatcher", "EvidencePolicy"),
        "gold_lost_after_recall": ("候选去向记录", "融合及重排 TopK", "最终证据门禁"),
        "suspected_gold": ("核对两份资料的技术事实差异", "人工审核是否允许替代来源"),
        "no_defined_risk": (),
    }.get(code, ())
    return {
        "code": code,
        "title": title,
        "observed": observed,
        "case_id": case.case_id,
        "stage_from": stage_from,
        "stage_to": stage_to,
        "priority": priority,
        "chunk_ids": list(dict.fromkeys(str(item) for item in ids if item)),
        "inspection_targets": list(targets),
        "confidence": "observed",
    }


def _stage_summary(
    case: GoldCase, stage_payload: dict[str, Any], config: ReportConfig,
    official: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = stage_payload.get("candidates", [])
    positives = {str(item.get("evidence_id")) for item in _evidence_entries(case, "positive")}
    found: set[str] = set()
    negatives: set[str] = set()
    ranks: list[int] = []
    for candidate in candidates:
        rank = candidate.get("stage_rank")
        if rank is None:
            continue
        for match in candidate.get("matches", []):
            if match.get("status") in {"exact", "high"} or (config.count_partial_matches and match.get("status") == "partial"):
                if match.get("kind") == "positive":
                    found.add(str(match.get("evidence_id")))
                    ranks.append(int(rank))
                elif match.get("kind") == "hard_negative":
                    negatives.add(str(match.get("evidence_id")))
    first = min(ranks) if ranks else None
    result = {
        "executed": bool(stage_payload.get("executed")),
        "candidate_count": len(candidates),
        # 拒绝项仍留在明细供排查，但没有阶段名次，不能充当热力图分母。
        "ranked_candidate_count": sum(item.get("stage_rank") is not None for item in candidates),
        "gold_found": len(found),
        "gold_total": len(positives),
        "gold_coverage": len(found) / len(positives) if positives else None,
        "first_gold_rank": first,
        "hit_at_1": bool(first is not None and first <= 1),
        "hit_at_5": bool(first is not None and first <= 5),
        "mrr": 1.0 / first if first else 0.0 if positives else None,
        "hard_negative_found": len(negatives),
        "hard_negative_first_rank": min(
            int(candidate.get("stage_rank"))
            for candidate in candidates
            if candidate.get("stage_rank") is not None
            for match in candidate.get("matches", [])
            if match.get("kind") == "hard_negative"
            and (match.get("status") in {"exact", "high"} or (config.count_partial_matches and match.get("status") == "partial"))
        ) if negatives else None,
    }
    if official and "gold_completion_ranks" in official:
        first = official.get("first_gold_rank")
        result.update({
            "gold_found": official.get("gold_evidence_found", 0),
            "gold_total": official.get("gold_evidence_total_declared", len(positives)),
            "gold_coverage": official.get("gold_evidence_recall"),
            "first_gold_rank": first,
            "first_gold_fragment_rank": official.get("first_gold_fragment_rank"),
            "gold_completion_ranks": official.get("gold_completion_ranks", {}),
            "gold_match_types": official.get("gold_match_types", {}),
            "hit_at_1": first is not None and first <= 1,
            "hit_at_5": first is not None and first <= 5,
            "mrr": official.get("reciprocal_rank"),
        })
    return result


def _evidence_exit(
    evidence: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    raw_result: dict[str, Any],
) -> dict[str, Any]:
    """指出一条标注证据首次从主链路退出的阶段及原始处理原因。"""

    evidence_id = str(evidence["evidence_id"])
    ranks = evidence.get("stages", {})
    first = next((stage for stage in STAGE_ORDER if ranks.get(stage)), None)
    if first is None:
        mapped = next((row for row in raw_result.get("gold_mappings", [])
                       if row.get("evidence_id") == evidence_id), {})
        suspected_ids = set(map(str, mapped.get("suspected_chunk_ids", [])))
        reason = (
            "存在跨来源相同引文，但整块正文不等价；已标疑似金标，未自动计分"
            if suspected_ids and any(
                str(candidate.get("chunk_id") or "") in suspected_ids
                for stage in ("lexical", "vector", "fusion")
                for candidate in stages.get(stage, {}).get("candidates", [])
            ) else
            "Gold 引文未能在本次索引中定位"
            if mapped and not mapped.get("chunk_ids") else
            "关键词与向量的已记录 Top-K 内均未出现"
        )
        return {"from": None, "to": None, "reason": reason, "status": "not_in_recorded_recall"}

    # 关键词与向量是并行入口；从融合阶段起才是一条连续的候选链。
    if first in {"lexical", "vector"} and not ranks.get("fusion"):
        return {"from": first, "to": "fusion", "reason": "未进入已记录的融合候选深度", "status": "fusion_cutoff"}
    chain = STAGE_ORDER[STAGE_ORDER.index("fusion"):] if ranks.get("fusion") else STAGE_ORDER[STAGE_ORDER.index(first):]
    mapped = next((row for row in raw_result.get("gold_mappings", [])
                   if row.get("evidence_id") == evidence_id), {})
    mapped_ids = set(map(str, mapped.get("chunk_ids", [])))
    for current, following in zip(chain, chain[1:]):
        if not ranks.get(current) or ranks.get(following):
            continue
        transition = next((item for item in raw_result.get("candidate_transitions", [])
                           if item.get("from_stage") == current and item.get("to_stage") == following
                           and str(item.get("chunk_id") or "") in mapped_ids
                           and item.get("status") == "removed"), None)
        if transition:
            return {
                "from": current, "to": following,
                "reason": str(transition.get("reason") or "未记录原因"),
                "status": str(transition.get("reason_code") or "unknown_legacy_result"),
                "chunk_id": str(transition.get("chunk_id") or ""),
            }
        candidates = [
            item for item in stages.get(current, {}).get("candidates", [])
            if any(match.get("evidence_id") == evidence_id and match.get("status") in {"exact", "high"}
                   for match in item.get("matches", []))
        ]
        chunk_ids = {str(item.get("chunk_id") or "") for item in candidates}
        reason = "未进入下一阶段已记录的候选深度"
        status = "stage_cutoff"
        if following == "deduplicated":
            dispositions = [
                row for event in raw_result.get("events", []) if isinstance(event, dict)
                for row in event.get("candidate_dispositions", []) if isinstance(row, dict)
                and str(row.get("chunk_id") or "") in chunk_ids
            ]
            if dispositions:
                chosen = dispositions[0]
                status = str(chosen.get("status") or status)
                reason = str(chosen.get("reason") or reason)
                if chosen.get("representative_chunk_id"):
                    reason += f"；代表正文 {chosen['representative_chunk_id']}"
        elif following == "rerank":
            reason = "超过重排输入 Top-K，未获得 Cross-Encoder 分数"
        elif following in {"constraints", "evidence"}:
            next_rows = stages.get(following, {}).get("candidates", [])
            rejected = next((item for item in next_rows if str(item.get("chunk_id") or "") in chunk_ids), None)
            if rejected:
                reason = str(rejected.get("reason") or (rejected.get("constraint_decision") or {}).get("reason") or reason)
                status = "rejected"
        elif following == "context":
            reason = "已入选最终证据，但未纳入实际上下文；检查字符预算或防御性合并"
            status = "context_budget_or_merge"
        return {"from": current, "to": following, "reason": reason, "status": status}
    return {"from": None, "to": None, "reason": "已到达最后一个执行阶段", "status": "survived"}


def _query_record(raw_result: dict[str, Any]) -> dict[str, Any]:
    events = [item for item in raw_result.get("events", []) if isinstance(item, dict)]
    planned = next((item.get("query_plan") for item in events if isinstance(item.get("query_plan"), dict)), {})
    executed = next((item for item in events if item.get("lexical_fts_expression") is not None), {})
    limits = next((item.get("query_limits") for item in events if isinstance(item.get("query_limits"), dict)), {})
    lexical_routes = next((item.get("lexical_route_diagnostics") for item in events
                           if isinstance(item.get("lexical_route_diagnostics"), list)), [])
    vector_routes = next((item.get("vector_route_diagnostics") for item in events
                          if isinstance(item.get("vector_route_diagnostics"), list)), [])
    return {
        "original_text": planned.get("original_text") or raw_result.get("question", ""),
        "lexical_terms": (
            lexical_routes[0].get("actual_terms", [])
            if lexical_routes and isinstance(lexical_routes[0], dict)
            and isinstance(lexical_routes[0].get("actual_terms"), list)
            else executed.get("lexical_terms", planned.get("lexical_terms", []))
        ),
        "legacy_lexical_terms": planned.get("lexical_terms", []),
        "lexical_backend": (
            lexical_routes[0].get("backend", "legacy")
            if lexical_routes and isinstance(lexical_routes[0], dict) else "legacy"
        ),
        "lexical_expression": executed.get("lexical_fts_expression", planned.get("lexical_expression", "")),
        "semantic_query": planned.get("semantic_query", raw_result.get("question", "")),
        "spans": planned.get("spans", []),
        "transformations": planned.get("transformations", []),
        "lexical_variants": planned.get("lexical_variants", []),
        "semantic_variants": planned.get("semantic_variants", []),
        "executed_lexical_routes": lexical_routes,
        "executed_vector_routes": vector_routes,
        "constraints": planned.get("constraints", []),
        "limits": limits,
        "analyzer_version": planned.get("analyzer_version", ""),
    }


def _generation_record(raw_result: dict[str, Any]) -> dict[str, Any]:
    """保留生成初稿和引用预检结果，失败题也能解释模型实际返回了什么。"""

    drafts = [
        event for event in raw_result.get("events", [])
        if isinstance(event, dict) and event.get("stage") == "draft_generated"
    ]
    draft = drafts[-1] if drafts else {}
    return {
        "draft_answer": str(draft.get("draft_answer") or ""),
        "citation_check": _json_safe(draft.get("citation_check") or {}),
        "failed_stage": str(raw_result.get("failed_stage") or ""),
    }


def build_report_detail(
    dataset: GoldDataset,
    run_metadata: dict[str, Any],
    results: list[dict[str, Any]],
    config: ReportConfig | None = None,
    *,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or ReportConfig()
    result_by_id = {str(item.get("id") or ""): item for item in results}
    validation = _snapshot_validation(dataset, run_metadata, results, config)
    cases: list[dict[str, Any]] = []
    for case in dataset.cases:
        raw_result = result_by_id.get(case.case_id)
        if raw_result is None:
            cases.append({
                "id": case.case_id,
                "group_id": case.group_id,
                "question": case.question,
                "expected_behavior": case.expected_behavior,
                "review_status": case.review_status,
                "score_eligible": False,
                "excluded_reason": "未运行或未出现在逐题结果中",
                "diagnostics": [_diag("not_run", "未运行", "没有找到本次运行的逐题结果。", case, None, None, "high", ())],
            })
            continue
        stage_data: dict[str, dict[str, Any]] = {}
        for stage in STAGE_ORDER:
            raw_candidates = raw_result.get("candidate_stages", {}).get(stage, [])
            if stage == "context":
                # 旧运行的 context 可能由扁平化 SourceRef 生成，需回到实际
                # ContextBuilder 事件按代表正文重建；来源不单独占排名。
                context_event = next(
                    (event for event in reversed(raw_result.get("events", []))
                     if isinstance(event, dict) and (event.get("context_entries") or event.get("context_sources"))),
                    None,
                )
                if context_event:
                    entries = context_event.get("context_entries") or context_event.get("context_sources") or []
                    selected = {
                        str(item.get("chunk_id") or ""): item
                        for item in raw_result.get("candidate_stages", {}).get("evidence", [])
                        if isinstance(item, dict) and item.get("selected") is True
                    }
                    raw_candidates = [
                        {**selected[str(entry.get("chunk_id") or "")],
                         "included_text": entry.get("included_text"),
                         "truncated": bool(entry.get("truncated")),
                         "context_reason": entry.get("reason", "")}
                        for entry in entries if isinstance(entry, dict)
                        and str(entry.get("chunk_id") or "") in selected
                    ]
            mappings = raw_result.get("gold_mappings", [])
            formal = mappings if any("declared_chunk_ids" in item for item in mappings) else None
            normalized, _ = _match_stage_candidates(case, stage, list(raw_candidates), config, formal)
            stage_data[stage] = {
                "executed": _stage_executed(raw_result, stage),
                "candidates": normalized,
            }
        rank_history: dict[str, list[dict[str, Any]]] = {}
        for stage in STAGE_ORDER:
            for candidate in stage_data[stage]["candidates"]:
                group_id = str(candidate.get("content_group_id") or candidate.get("chunk_id") or "")
                if group_id and candidate.get("stage_rank") is not None:
                    rank_history.setdefault(group_id, []).append({
                        "stage": stage,
                        "label": STAGE_LABELS.get(stage, stage),
                        "rank": candidate.get("stage_rank"),
                    })
        for payload in stage_data.values():
            for candidate in payload["candidates"]:
                candidate["rank_history"] = rank_history.get(
                    str(candidate.get("content_group_id") or candidate.get("chunk_id") or ""), []
                )
        # 上下文事件包含实际纳入的正文；用它覆盖默认的完整候选正文。
        context_entries = [
            entry for event in raw_result.get("events", []) if isinstance(event, dict)
            for entry in (event.get("context_sources") or []) if isinstance(entry, dict)
        ]
        if context_entries:
            by_id = {str(item.get("chunk_id") or ""): item for item in stage_data["context"]["candidates"]}
            for entry in context_entries:
                item = by_id.get(str(entry.get("chunk_id") or ""))
                if item is not None:
                    item["included_text"] = entry.get("included_text")
                    item["truncated"] = bool(entry.get("truncated"))
                    item["context_reason"] = entry.get("reason")
        summaries = {
            stage: _stage_summary(case, payload, config, raw_result.get("stages", {}).get(stage))
            for stage, payload in stage_data.items()
        }
        positives = _evidence_entries(case, "positive")
        negatives = _evidence_entries(case, "hard_negative")
        for evidence in positives + negatives:
            mapped = next((row for row in raw_result.get("gold_mappings", [])
                           if row.get("evidence_id") == evidence["evidence_id"]), {})
            evidence["mapping"] = _json_safe(mapped)
            suspected_ids = set(map(str, mapped.get("suspected_chunk_ids", [])))
            evidence["suspected_stages"] = {
                stage: [int(candidate["stage_rank"]) for candidate in payload["candidates"]
                        if candidate.get("stage_rank") is not None
                        and str(candidate.get("chunk_id") or "") in suspected_ids]
                for stage, payload in stage_data.items()
            }
            evidence["stages"] = {}
            for stage, payload in stage_data.items():
                evidence["stages"][stage] = sorted({
                    int(candidate["stage_rank"])
                    for candidate in payload["candidates"]
                    if candidate.get("stage_rank") is not None
                    for match in candidate.get("matches", [])
                    if match.get("evidence_id") == evidence["evidence_id"]
                    and (match.get("status") in {"exact", "high"} or (config.count_partial_matches and match.get("status") == "partial"))
                })
                if evidence["kind"] == "positive" and "gold_completion_ranks" in summaries[stage]:
                    complete_rank = summaries[stage]["gold_completion_ranks"].get(evidence["evidence_id"])
                    evidence["stages"][stage] = [int(complete_rank)] if complete_rank is not None else []
            evidence["first_rank"] = min(
                (rank for values in evidence["stages"].values() for rank in values),
                default=None,
            )
            evidence["exit"] = _evidence_exit(evidence, stage_data, raw_result)
        diagnostics = _case_diagnostics(case, stage_data, config, raw_result.get("stages", {}))
        gold_exit = next((item["exit"] for item in positives
                          if item["exit"]["status"] not in {"survived", "not_in_recorded_recall"}), None)
        if gold_exit:
            for diagnostic in diagnostics:
                if diagnostic["code"] == "gold_lost_after_recall":
                    diagnostic["observed"] = (
                        f"金标从 {STAGE_LABELS.get(gold_exit['from'], gold_exit['from'])} "
                        f"到 {STAGE_LABELS.get(gold_exit['to'], gold_exit['to'])} 退出：{gold_exit['reason']}"
                    )
        eligible = case.review_status == "reviewed" and case.expected_behavior != "refuse" and not raw_result.get("error")
        if validation["mismatch"] and not config.allow_snapshot_mismatch:
            eligible = False
        cases.append({
            "id": case.case_id,
            "case_index": raw_result.get("case_index"),
            "group_id": case.group_id,
            "question": case.question,
            "expected_behavior": case.expected_behavior,
            "actual_behavior": raw_result.get("actual_behavior", "not_run"),
            "review_status": case.review_status,
            "constraints": _json_safe(case.constraints),
            "missing_fields": list(case.missing_fields),
            "required_facts": list(case.required_facts),
            "answer_checks": _json_safe(case.answer_checks),
            "acceptable_alternative_evidence": _json_safe(case.raw.get("acceptable_alternative_evidence") or []),
            "forbidden_claims": list(case.forbidden_claims),
            "gold_evidence": positives,
            "hard_negatives": negatives,
            "query": _query_record(raw_result),
            "stage_data": stage_data,
            "candidate_transitions": _json_safe(raw_result.get("candidate_transitions") or []),
            "stage_summaries": summaries,
            "diagnostics": diagnostics,
            "answer": _json_safe(raw_result.get("answer") or {}),
            "gold_judgement": _json_safe(raw_result.get("gold_judgement") or {}),
            "generation": _generation_record(raw_result),
            "timing": case_timing(raw_result),
            "error": _json_safe(raw_result.get("error")),
            "score_eligible": eligible,
            "excluded_reason": None if eligible else (
                "拒答题不计入正证据指标" if case.expected_behavior == "refuse" else
                "索引版本不一致" if validation["mismatch"] and not config.allow_snapshot_mismatch else
                f"review_status={case.review_status}" if case.review_status != "reviewed" else
                "运行失败或未完成"
            ),
        })

    scored = [item for item in cases if item.get("score_eligible")]
    route_hits: dict[str, int] = {}
    loss_reasons: dict[str, int] = {}
    evidence_types = {"指定金标": 0, "排版等价金标": 0, "疑似金标": 0, "指定负例": 0}
    for item in cases:
        if item.get("error"):
            continue
        gold_ids = {
            str(chunk_id)
            for evidence in item.get("gold_evidence", [])
            for chunk_id in (evidence.get("mapping") or {}).get("chunk_ids", [])
        }
        per_case_routes: set[str] = set()
        for stage, field in (("lexical", "lexical_route_hits"), ("vector", "vector_route_hits")):
            for candidate in item.get("stage_data", {}).get(stage, {}).get("candidates", []):
                if str(candidate.get("chunk_id") or "") not in gold_ids:
                    continue
                per_case_routes.update(str(hit.get("route_id") or "") for hit in candidate.get(field, []))
        for route_id in per_case_routes:
            if route_id:
                route_hits[route_id] = route_hits.get(route_id, 0) + 1
        for transition in item.get("candidate_transitions", []):
            if (transition.get("chunk_id") in gold_ids
                    and transition.get("status") == "removed"
                    and transition.get("from_stage") not in {"lexical", "vector"}):
                code = str(transition.get("reason_code") or "unknown_legacy_result")
                loss_reasons[code] = loss_reasons.get(code, 0) + 1
        for evidence in item.get("gold_evidence", []):
            mapping = evidence.get("mapping") or {}
            for key, label in (("declared_chunk_ids", "指定金标"),
                               ("equivalent_chunk_ids", "排版等价金标"),
                               ("suspected_chunk_ids", "疑似金标")):
                evidence_types[label] += len(mapping.get(key, []))
        for evidence in item.get("hard_negatives", []):
            evidence_types["指定负例"] += len((evidence.get("mapping") or {}).get("chunk_ids", []))
    summary: dict[str, Any] = {
        "case_count": len(cases),
        "scored_case_count": len(scored),
        "reviewed_case_count": sum(item.get("review_status") == "reviewed" for item in cases),
        "excluded_case_count": len(cases) - len(scored),
        "refusal_case_count": sum(item.get("expected_behavior") == "refuse" for item in cases),
        "diagnostic_counts": {},
        "stages": {},
        "timing": run_timing(run_metadata, results),
        "diagnostic_overview": {
            "routeHits": route_hits,
            "lossReasons": loss_reasons,
            "evidenceTypes": evidence_types,
        },
        "answer_judgement_counts": {
            verdict: sum(item.get("gold_judgement", {}).get("verdict") == verdict for item in cases)
            for verdict in ("consistent", "incomplete", "safety_gap", "needs_clarification",
                            "behavior_mismatch", "incorrect", "uncertain", "unavailable", "inconsistent")
        },
    }
    for item in cases:
        for diagnostic in item.get("diagnostics", []):
            code = str(diagnostic.get("code") or "")
            summary["diagnostic_counts"][code] = summary["diagnostic_counts"].get(code, 0) + 1
    for stage in STAGE_ORDER:
        rows = [item["stage_summaries"][stage] for item in scored if stage in item.get("stage_summaries", {})]
        first_ranks = [row["first_gold_rank"] for row in rows if row.get("first_gold_rank") is not None]
        recalls = [row["gold_coverage"] for row in rows if row.get("gold_coverage") is not None]
        mrrs = [row["mrr"] for row in rows if row.get("mrr") is not None]
        summary["stages"][stage] = {
            "label": STAGE_LABELS.get(stage, stage),
            "cases": len(rows),
            "hit_at_1_count": sum(bool(row.get("hit_at_1")) for row in rows),
            "hit_at_5_count": sum(bool(row.get("hit_at_5")) for row in rows),
            "hit_at_1": sum(bool(row.get("hit_at_1")) for row in rows) / len(rows) if rows else None,
            "hit_at_5": sum(bool(row.get("hit_at_5")) for row in rows) / len(rows) if rows else None,
            "mean_coverage": sum(recalls) / len(recalls) if recalls else None,
            "mrr": sum(mrrs) / len(mrrs) if mrrs else None,
            "mean_first_gold_rank": sum(first_ranks) / len(first_ranks) if first_ranks else None,
            "hard_negative_count": sum(int(row.get("hard_negative_found") or 0) for row in rows),
        }
    detail: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "matcher_version": REPORT_MATCHER_VERSION,
        "metric_version": REPORT_METRIC_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": _json_safe(run_metadata),
        "config": asdict(config),
        "validation": validation,
        "answer_rubric": {
            "version": "answer-checks-v2" if any("answer_checks" in case.raw for case in dataset.cases)
                       else "legacy-facts-v1",
            "review_note": str((dataset.metadata.get("_阅读说明") or {}).get("审核提示") or ""),
        },
        "summary": summary,
        "cases": cases,
        "metric_definitions": {
            "Hit@1": "非拒答且已审核的计分题中，首条完整金标证据完成排名 ≤ 1 的题数 ÷ 计分题数。",
            "Hit@5": "非拒答且已审核的计分题中，首条完整金标证据完成排名 ≤ 5 的题数 ÷ 计分题数。",
            "MRR": "每题取首条完整金标证据完成排名的倒数，再对计分题求平均；未命中贡献 0。",
            "证据覆盖率": "本题完整找到的金标条目数 ÷ 本题声明金标总数；索引未定位的 Gold 仍计入分母，拒答题不计正证据。",
            "疑似金标": "相似或同名跨格式来源存在技术字符差异时保留排名供复核，不计入正式指标。",
            "相似未标注": "没有匹配当前 Gold 或 hard negative 的候选，不代表语义错误。",
        },
        "baseline": _compare_baseline(detail_stub=None, current_cases=cases, baseline=baseline) if baseline else {},
    }
    return detail


def _compare_baseline(detail_stub: Any, current_cases: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    old = {str(item.get("id")): item for item in baseline.get("cases", [])}
    changes: dict[str, dict[str, Any]] = {}
    for item in current_cases:
        previous = old.get(str(item.get("id")))
        if previous is None:
            continue
        current_summary = item.get("stage_summaries", {}).get("context", {})
        previous_summary = previous.get("stage_summaries", {}).get("context", {})
        current_hit = bool(current_summary.get("hit_at_5")) if item.get("score_eligible") else None
        previous_hit = bool(previous_summary.get("hit_at_5")) if previous.get("score_eligible") else None
        state = "unchanged"
        if previous_hit is not None and current_hit is not None:
            if previous_hit and not current_hit:
                state = "对→错"
            elif not previous_hit and current_hit:
                state = "错→对"
        changes[str(item.get("id"))] = {
            "state": state,
            "previous_hit_at_5": previous_hit,
            "current_hit_at_5": current_hit,
            "coverage_delta": (current_summary.get("gold_coverage") or 0) - (previous_summary.get("gold_coverage") or 0),
            "mrr_delta": (current_summary.get("mrr") or 0) - (previous_summary.get("mrr") or 0),
        }
    return {"available": True, "case_changes": changes}


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def write_report_files(detail: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report_detail.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "id", "question", "expected_behavior", "review_status", "score_eligible",
        "coverage", "coverage_total", "first_gold_rank", "hard_negative_first_rank",
        "diagnostics", "actual_keywords", "lexical_expression", "answer_verdict",
        "answer_findings", "baseline_change",
    ]
    with (output_dir / "report_cases.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in detail.get("cases", []):
            summary = case.get("stage_summaries", {}).get(detail.get("config", {}).get("default_stage", "context"), {})
            change = detail.get("baseline", {}).get("case_changes", {}).get(str(case.get("id")), {})
            writer.writerow({
                "id": case.get("id"),
                "question": case.get("question"),
                "expected_behavior": case.get("expected_behavior"),
                "review_status": case.get("review_status"),
                "score_eligible": case.get("score_eligible"),
                "coverage": summary.get("gold_found"),
                "coverage_total": summary.get("gold_total"),
                "first_gold_rank": summary.get("first_gold_rank"),
                "hard_negative_first_rank": summary.get("hard_negative_first_rank"),
                "diagnostics": [item.get("title") for item in case.get("diagnostics", [])],
                "actual_keywords": case.get("query", {}).get("lexical_terms", []),
                "lexical_expression": case.get("query", {}).get("lexical_expression", ""),
                "answer_verdict": case.get("gold_judgement", {}).get("verdict", ""),
                "answer_findings": case.get("gold_judgement", {}).get("findings", []),
                "baseline_change": change.get("state", ""),
            })
    (output_dir / "report.html").write_text(render_html(detail), encoding="utf-8")


def render_html(detail: dict[str, Any]) -> str:
    payload = json.dumps(detail, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")
    title = "RAG 检索诊断报告"
    html_text = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#667085;--line:#e1e6ef;--green:#2e7d32;--green-bg:#e8f5e9;--red:#c62828;--red-bg:#ffebee;--yellow:#f9a825;--yellow-bg:#fffde7;--blue:#2563eb;--shadow:0 8px 24px rgba(30,48,80,.08)}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 "Microsoft YaHei","Segoe UI",sans-serif}}
.top{{position:sticky;top:0;z-index:5;background:#172033;color:#fff;padding:18px 28px;box-shadow:0 2px 12px #0002}} .top h1{{margin:0 0 3px;font-size:22px}} .top small{{color:#ccd5e3}} main{{max-width:1500px;margin:0 auto;padding:24px}} .banner{{padding:14px 18px;border-radius:12px;margin-bottom:16px;background:var(--green-bg);border:1px solid #a5d6a7}} .banner.bad{{background:var(--red-bg);border-color:#ef9a9a;color:#8b1b1b}} .legend{{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}} .legend span,.badge{{display:inline-flex;align-items:center;gap:5px;border-radius:999px;padding:3px 9px;font-size:12px;font-weight:600}} .gold{{color:var(--green);background:var(--green-bg)}} .negative{{color:var(--red);background:var(--red-bg)}} .mixed{{color:#6a1b9a;background:linear-gradient(90deg,var(--green-bg),var(--red-bg));border-left:4px solid var(--green);border-right:4px solid var(--red)}} .unlabelled{{color:#8a6500;background:var(--yellow-bg)}} .partial{{color:#8a6500;background:#fff8cf}} .muted{{color:var(--muted)}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}} .card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:var(--shadow)}} .card strong{{display:block;font-size:25px;margin:4px 0}} .card small{{color:var(--muted)}} .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .panel{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:var(--shadow);margin:16px 0}} .panel h2{{margin:0 0 12px;font-size:17px}} .priority{{display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:start;padding:12px;border-top:1px solid var(--line);cursor:pointer}} .priority:first-of-type{{border-top:0}} .priority .p{{font-weight:700;color:#b42318}} .priority .target{{color:var(--blue);font-size:12px}} .toolbar{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}} input,select,button{{font:inherit;border:1px solid #cbd5e1;border-radius:8px;padding:8px 10px;background:#fff;color:var(--ink)}} button{{cursor:pointer;background:#f8fafc}} button:hover{{border-color:var(--blue)}} .chart{{min-height:250px;overflow:auto}} svg text{{font-family:inherit;fill:var(--muted);font-size:11px}} .heatmap{{overflow:auto}} .heatmap table{{min-width:900px}} table{{border-collapse:collapse;width:100%}} th,td{{border-bottom:1px solid var(--line);padding:9px 8px;text-align:left;vertical-align:top}} th{{color:var(--muted);font-size:12px;white-space:nowrap;position:sticky;top:0;background:#fff}} tr.clickable{{cursor:pointer}} tr.clickable:hover{{background:#f8fbff}} .heat{{text-align:center;font-weight:700;border-radius:6px;padding:5px;min-width:60px}} .heat.good{{background:var(--green-bg);color:var(--green)}} .heat.warn{{background:#fff8cf;color:#8a6500}} .heat.bad{{background:var(--red-bg);color:var(--red)}} .case{{border:1px solid var(--line);border-radius:12px;margin:12px 0;overflow:hidden;background:#fff}} .case>summary{{padding:14px 16px;cursor:pointer;list-style:none}} .case>summary::-webkit-details-marker{{display:none}} .case>summary:hover{{background:#f8fbff}} .case-body{{padding:0 16px 18px}} .detail-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .evidence{{border-left:4px solid #cbd5e1;padding:8px 10px;margin:7px 0;background:#fafbfc}} .evidence.ok{{border-color:var(--green)}} .evidence.neg{{border-color:var(--red)}} .evidence.miss{{border-style:dashed}} .stage-block{{border-top:1px solid var(--line);padding:14px 0}} .stage-block h4{{margin:0 0 8px}} .candidate{{border:1px solid var(--line);border-left:5px solid #cbd5e1;border-radius:9px;padding:10px;margin:8px 0}} .candidate.gold{{border-left-color:var(--green)}} .candidate.negative{{border-left-color:var(--red)}} .candidate.mixed{{border-left-color:var(--green);border-right:5px solid var(--red)}} .candidate.unlabelled{{border-left-color:var(--yellow)}} .candidate-head{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}} .rank{{font-size:18px;font-weight:700}} .score{{color:var(--muted);font-size:12px}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc;border-radius:8px;padding:10px;margin:8px 0}} .hl-gold{{background:#b7dfba}} .hl-negative{{background:#ffc2c2}} .hl-mixed{{background:linear-gradient(90deg,#b7dfba,#ffc2c2)}} .filter-note{{color:var(--muted);margin:8px 0}} .pill{{font-size:11px;border-radius:5px;padding:2px 6px;background:#eef2f7;color:#475467}} .pill.high{{background:#ffe4e6;color:#9f1239}} .pill.medium{{background:#fff7cc;color:#8a6500}} .footer{{color:var(--muted);font-size:12px;padding:20px 0}} @media(max-width:900px){{.grid2,.detail-grid{{grid-template-columns:1fr}} main{{padding:14px}}}}
</style><style>.hl-partial-gold{{background:#fff1a8}} .hl-partial-negative{{background:#ffd6a3}} .candidate.partial_gold{{border-left-color:#f9a825;background:#fffdf7}} .candidate.partial_negative{{border-left-color:#f9a825;background:#fffaf0}} .expand-stage{{margin:4px 0 10px;color:#2563eb}}</style></head><body>
<header class="top"><h1>{title}</h1><small id="runLine"></small><div class="legend"><span class="gold">绿色：金标证据</span><span class="negative">红色：指定负例</span><span class="unlabelled">黄色：相似未标注</span><span class="mixed">双色：同块混合</span><span class="muted">灰色虚线：金标未召回</span></div></header>
<main><div id="banner"></div><section id="summary"></section><section class="panel"><h2>整体耗时与模块消耗</h2><div id="timingOverview"></div></section><section class="panel"><h2>优先排查清单</h2><div id="priority"></div></section><div class="grid2"><section class="panel"><h2>阶段趋势</h2><div class="toolbar"><select id="trendMetric"><option value="mrr">MRR</option><option value="mean_coverage">证据覆盖率</option><option value="hit_at_5">Hit@5</option></select></div><div class="chart" id="trend"></div></section><section class="panel"><h2>诊断标签分布</h2><div class="chart" id="diagnosticBars"></div></section></div><section class="panel"><h2>题目 × 阶段热力图</h2><div class="toolbar"><select id="heatMetric"><option value="first_gold_rank">首条金标名次</option><option value="gold_coverage">证据覆盖率</option></select><span class="filter-note">名次/本阶段实际有排名的候选数；右侧答案列由独立 LLM 裁判，未执行时显示“未判定”。点击格子打开题目。</span></div><div class="heatmap" id="heatmap"></div></section><section class="panel"><h2>题目总表</h2><div class="toolbar"><input id="search" placeholder="搜索题号、问题或文件名"><select id="behavior"><option value="all">全部预期行为</option><option value="answer">answer</option><option value="partial_answer">partial_answer</option><option value="clarify">clarify</option><option value="refuse">refuse</option></select><select id="diagnostic"><option value="all">全部诊断</option></select><select id="sort"><option value="priority">优先级</option><option value="id">题号</option><option value="coverage">证据覆盖</option><option value="rank">首条金标排名</option></select><button id="clear">清除筛选</button></div><div id="caseTable"></div></section><section class="panel"><h2>逐题详情</h2><div id="caseDetails"></div></section><section class="panel"><h2>未计分题与指标说明</h2><div id="excluded"></div><div id="definitions"></div></section><div class="footer">报告数据与图表内嵌在本文件中，可断网双击打开。颜色同时配有文字标签，不依赖颜色判断。</div></main>
<script>
const DATA={payload};
const STAGES={json.dumps(STAGE_LABELS, ensure_ascii=False)};
const ORDER={json.dumps(STAGE_ORDER, ensure_ascii=False)};
const $=id=>document.getElementById(id);
const esc=v=>String(v??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
const pct=v=>v==null?'—':(Number(v)*100).toFixed(1)+'%';
const rank=v=>v==null?'未命中':'#'+v;
const caseById=id=>DATA.cases.find(c=>c.id===id);
const diagLabel=d=>d?.title||d?.code||'';
function priorityScore(c){{return Math.max(...(c.diagnostics||[]).map(d=>d.priority==='high'?3:d.priority==='medium'?2:1),0)}}
function renderSummary(){{const s=DATA.summary, stage=DATA.config.default_stage, m=s.stages[stage]||{{}}, j=s.answer_judgement_counts||{{}}; const judged=Object.entries(j).filter(([name])=>name!=='unavailable').reduce((n,[,count])=>n+count,0); const action=(j.safety_gap||0)+(j.incomplete||0)+(j.needs_clarification||0)+(j.behavior_mismatch||0)+(j.incorrect||0)+(j.inconsistent||0); $('runLine').textContent=`运行 ${{DATA.run?.run_id||'未命名'}} · 模式 ${{DATA.run?.mode||'未知'}} · 默认阶段 ${{STAGES[stage]||stage}}`;
 $('summary').innerHTML=`<div class="cards"><div class="card"><small>题目总数</small><strong>${{s.case_count}}</strong><small>已审核 ${{s.reviewed_case_count}}</small></div><div class="card"><small>计分题</small><strong>${{s.scored_case_count}}</strong><small>拒答题不计正证据</small></div><div class="card"><small>默认阶段 Hit@5</small><strong>${{pct(m.hit_at_5)}}</strong><small>${{m.hit_at_5_count??0}}/${{m.cases??0}} · ${{STAGES[stage]||stage}}</small></div><div class="card"><small>默认阶段 MRR</small><strong>${{pct(m.mrr)}}</strong><small>首条金标名次倒数平均</small></div><div class="card"><small>平均证据覆盖</small><strong>${{pct(m.mean_coverage)}}</strong><small>按题宏平均</small></div><div class="card"><small>已定义风险题</small><strong>${{DATA.cases.filter(c=>(c.diagnostics||[]).some(d=>d.code!=='no_defined_risk')).length}}</strong><small>点击下方清单定位</small></div><div class="card"><small>需处理的答案</small><strong>${{judged?action:'—'}}</strong><small>${{judged?`安全 ${{j.safety_gap||0}} · 必答 ${{j.incomplete||0}} · 澄清 ${{j.needs_clarification||0}} · 其他 ${{(j.behavior_mismatch||0)+(j.incorrect||0)+(j.inconsistent||0)}}`:'未运行答案裁判'}}</small></div></div>`}}
function renderBanner(){{const v=DATA.validation; $('banner').className='banner'+(v.mismatch||v.missing?' bad':''); $('banner').innerHTML=(v.mismatch||v.missing)?`<b>索引版本校验警告</b>：${{esc(v.warning)}}${{v.allowed?'（已允许继续生成探索性报告，指标不作为同版本正式比较）':''}}`:`<b>索引版本已校验</b>：本次运行与 Gold 的 bundle_id 一致。`; if(DATA.run?.rejudge) $('banner').innerHTML+=`<div>本报告沿用原运行已保存的检索结果和答案，仅用新版金标判分项重新调用独立 LLM 裁判；重判耗时 ${{esc(DATA.run.rejudge.duration_ms)}} ms，不计入原运行模块耗时。</div>`;}}
function renderPriority(){{const rows=DATA.cases.flatMap(c=>(c.diagnostics||[]).filter(d=>d.code!=='no_defined_risk').map(d=>({{c,d}}))).sort((a,b)=>priorityScore(b.c)-priorityScore(a.c)); $('priority').innerHTML=rows.length?rows.slice(0,12).map(x=>`<div class="priority" data-case="${{esc(x.c.id)}}"><span class="pill ${{x.d.priority}}">${{esc(x.d.priority)}}</span><div><b>${{esc(x.c.id)}} · ${{esc(x.d.title)}}</b><div>${{esc(x.d.observed)}}</div><div class="target">建议查看：${{esc((x.d.inspection_targets||[]).join('、'))}}</div></div><span class="muted">→</span></div>`).join(''):'<div class="muted">没有发现已定义风险。</div>'; document.querySelectorAll('.priority').forEach(e=>e.onclick=()=>openCase(e.dataset.case))}}
function chartLine(metric){{const values=ORDER.map(s=>DATA.summary.stages[s]?.[metric]); const w=620,h=230,p=34; const nums=values.map(v=>v==null?0:Number(v)); const max=Math.max(1,...nums); const pts=nums.map((v,i)=>`${{p+i*(w-p*2)/Math.max(1,nums.length-1)}},${{h-p-(v/max)*(h-p*2)}}`).join(' '); return `<svg viewBox="0 0 ${{w}} ${{h}}" role="img" aria-label="阶段趋势"><line x1="${{p}}" y1="${{h-p}}" x2="${{w-p}}" y2="${{h-p}}" stroke="#cbd5e1"/><polyline points="${{pts}}" fill="none" stroke="#2563eb" stroke-width="3"/>${{nums.map((v,i)=>`<circle cx="${{p+i*(w-p*2)/Math.max(1,nums.length-1)}}" cy="${{h-p-(v/max)*(h-p*2)}}" r="5" fill="#2563eb"><title>${{STAGES[ORDER[i]]}}：${{pct(values[i])}}</title></circle><text x="${{p+i*(w-p*2)/Math.max(1,nums.length-1)}}" y="${{h-8}}" text-anchor="middle">${{STAGES[ORDER[i]].slice(0,4)}}</text>`).join('')}}</svg>`}}
function renderCharts(){{$('trend').innerHTML=chartLine($('trendMetric').value); const counts=Object.entries(DATA.summary.diagnostic_counts||{{}}).filter(([k])=>k!=='no_defined_risk').sort((a,b)=>b[1]-a[1]); const max=Math.max(1,...counts.map(x=>x[1])); $('diagnosticBars').innerHTML=counts.length?counts.map(([code,n])=>`<div style="display:grid;grid-template-columns:120px 1fr 35px;gap:8px;align-items:center;margin:9px 0;cursor:pointer" data-diagnostic="${{esc(code)}}"><span>${{esc((DATA.cases.flatMap(c=>c.diagnostics||[]).find(d=>d.code===code)||{{}}).title||code)}}</span><span style="height:16px;background:#e9eef5;border-radius:8px"><i style="display:block;width:${{n/max*100}}%;height:100%;background:#e45757;border-radius:8px"></i></span><b>${{n}}</b></div>`).join(''):'<span class="muted">没有风险标签</span>'; document.querySelectorAll('[data-diagnostic]').forEach(e=>e.onclick=()=>{{ $('diagnostic').value=e.dataset.diagnostic; renderCaseTable() }})}}
function renderHeatmap(){{
  const metric=$('heatMetric').value;
  let out='<table><thead><tr><th>题目</th>'+ORDER.map(s=>`<th>${{STAGES[s]}}</th>`).join('')+'<th>答案与金标</th></tr></thead><tbody>';
  DATA.cases.forEach(c=>{{
    out+=`<tr><th>${{esc(c.id)}}</th>`+ORDER.map(s=>{{
      const m=c.stage_summaries?.[s]||{{}};
      const value=m[metric];
      const total=m.ranked_candidate_count??(c.stage_data?.[s]?.candidates||[]).filter(x=>Number.isInteger(x.stage_rank)).length;
      const text=metric==='gold_coverage'?(value==null?'—':pct(value)):
        m.gold_total===0?'无金标':!m.executed?'未执行':value==null?`未命中/${{total}}`:`${{value}}/${{total}}`;
      const cls=metric==='gold_coverage'?(value==null?'':value>=.8?'good':value>0?'warn':'bad'):
        m.gold_total===0||!m.executed?'':value==null?'bad':value<=5?'good':value<=20?'warn':'bad';
      return `<td><div class="heat ${{cls}}" data-case="${{esc(c.id)}}">${{text}}</div></td>`;
    }}).join('')+(() => {{
      const j=c.gold_judgement||{{}};
      const labels={{consistent:'符合',incomplete:'必答缺失',safety_gap:'安全缺口',needs_clarification:'需澄清',behavior_mismatch:'方式不符',incorrect:'明确错误',uncertain:'待复核',unavailable:'未判定',inconsistent:'旧版不一致'}};
      const verdict=j.verdict||'unavailable';
      const cls=verdict==='consistent'?'good':verdict==='incorrect'||verdict==='inconsistent'?'bad':verdict==='unavailable'?'':'warn';
      const title=j.behavior_reason||j.reason||'本次没有可用的独立金标裁判结果';
      return `<td><div class="heat ${{cls}}" data-case="${{esc(c.id)}}" title="${{esc(title)}}">${{labels[verdict]||esc(verdict)}}</div></td>`;
    }})()+'</tr>';
  }});
  $('heatmap').innerHTML=out+'</tbody></table>';
  document.querySelectorAll('.heat[data-case]').forEach(e=>e.onclick=()=>openCase(e.dataset.case));
}}
function labels(c){{return (c.diagnostics||[]).map(d=>`<span class="pill ${{d.priority}}">${{esc(d.title)}}</span>`).join(' ')}}
function renderCaseTable(){{const q=$('search').value.trim().toLowerCase(), b=$('behavior').value, d=$('diagnostic').value, sort=$('sort').value, stage=DATA.config.default_stage; let rows=DATA.cases.filter(c=>{{const hay=[c.id,c.question,...(c.stage_data?.[stage]?.candidates||[]).map(x=>x.filename)].join(' ').toLowerCase(); return (!q||hay.includes(q))&&(b==='all'||c.expected_behavior===b)&&(d==='all'||(c.diagnostics||[]).some(x=>x.code===d))}}); rows.sort((a,b)=>sort==='id'?a.id.localeCompare(b.id):sort==='coverage'?((b.stage_summaries?.[stage]?.gold_coverage||0)-(a.stage_summaries?.[stage]?.gold_coverage||0)):sort==='rank'?((a.stage_summaries?.[stage]?.first_gold_rank||999)-(b.stage_summaries?.[stage]?.first_gold_rank||999)):priorityScore(b)-priorityScore(a)); $('caseTable').innerHTML=`<div class="filter-note">显示 ${{rows.length}} / ${{DATA.cases.length}} 题；点击行可展开详情。</div><table><thead><tr><th>题号</th><th>问题</th><th>预期</th><th>默认阶段证据覆盖</th><th>首条金标</th><th>首个负例</th><th>诊断</th></tr></thead><tbody>${{rows.map(c=>{{const m=c.stage_summaries?.[stage]||{{}}; return `<tr class="clickable" data-case="${{esc(c.id)}}"><td><b>${{esc(c.id)}}</b></td><td>${{esc(c.question.slice(0,70))}}</td><td>${{esc(c.expected_behavior)}}</td><td>${{m.gold_found??'—'}}/${{m.gold_total??'—'}}</td><td>${{rank(m.first_gold_rank)}}</td><td>${{rank(m.hard_negative_first_rank)}}</td><td>${{labels(c)}}</td></tr>`}}).join('')}}</tbody></table>`; document.querySelectorAll('#caseTable [data-case]').forEach(e=>e.onclick=()=>openCase(e.dataset.case))}}
function matchSummary(e){{return `<div class="evidence ${{e.kind==='positive'?'ok':'neg'}}"><b>${{e.evidence_id}}</b> · ${{esc(e.file||'')}} · 第 ${{esc(e.page||'?')}} 页<br>${{esc(e.quote||'')}}<br><span class="muted">各阶段名次：${{ORDER.map(s=>`${{STAGES[s]}} ${{(e.stages?.[s]||[]).map(x=>'#'+x).join('、')||'未出现'}}`).join('　')}}</span></div>`}}
function candidateCard(c,stage){{const category=c.category||'unlabelled'; const text=(c.included_text??c.text??''); const content=(c.highlight_segments||[]).map(s=>`<span class="hl-${{s.kinds.includes('positive')&&s.kinds.includes('hard_negative')?'mixed':s.kinds.includes('positive')?'gold':s.kinds.includes('hard_negative')?'negative':''}}">${{esc(s.text)}}</span>`).join('')||esc(text); const history=(c.rank_history||[]).map(h=>`${{STAGES[h.stage]||h.stage}} #${{h.rank}}`).join(' → '); return `<div class="candidate ${{category}}"><div class="candidate-head"><span class="rank">${{c.rank_display==='未入选'?'未入选':'#'+esc(c.rank_display)}}</span><b>${{esc(c.filename||'')}}</b><span class="pill">${{esc(c.location||'')}}</span><span class="badge ${{category}}">${{category==='gold'?'金标':category==='negative'?'指定负例':category==='mixed'?'同块混合':category==='unlabelled'?'相似未标注':category}}</span><span class="score">向量 ${{esc(c.vector_score??'—')}} · 融合 ${{esc(c.fused_score??'—')}} · 重排 ${{esc(c.rerank_score??'—')}} · 综合重排 ${{esc(c.rerank_score_combined??'—')}} · 直接回答支持 ${{esc(c.answer_support_score??'—')}}</span></div><div class="muted">状态：${{esc(c.status||'')}}；原因：${{esc(c.reason||c.constraint_decision?.reason||'')}}<br>排名轨迹：${{esc(history||'未进入后续阶段')}}</div><pre>${{content}}</pre></div>`}}
function detailHtml(c){{const stage=DATA.config.default_stage; return `<details class="case" id="case-${{esc(c.id)}}"><summary><b>${{esc(c.id)}}</b> · ${{esc(c.question)}}　${{labels(c)}}</summary><div class="case-body"><p><b>预期：</b>${{esc(c.expected_behavior)}}　<b>group：</b>${{esc(c.group_id)}}　<b>约束：</b>${{esc(JSON.stringify(c.constraints||{{}}))}}　<b>缺失条件：</b>${{esc((c.missing_fields||[]).join('、')||'无')}}</p><div class="detail-grid"><div><h3>期望区：金标证据</h3>${{(c.gold_evidence||[]).map(matchSummary).join('')||'<div class="muted">本题没有正向金标。</div>'}}<h3>负例区</h3>${{(c.hard_negatives||[]).map(matchSummary).join('')||'<div class="muted">无指定负例。</div>'}}</div><div><h3>排名轨迹说明</h3><p class="muted">每个候选在下方各阶段按真实记录显示名次；“未出现”不等于已确认被拒绝，若有原因会单独显示。</p><table><thead><tr><th>阶段</th><th>证据覆盖</th><th>首条金标</th><th>首个负例</th></tr></thead><tbody>${{ORDER.map(s=>{{const m=c.stage_summaries?.[s]||{{}}; return `<tr><td>${{STAGES[s]}}</td><td>${{m.gold_found??'—'}}/${{m.gold_total??'—'}}</td><td>${{rank(m.first_gold_rank)}}</td><td>${{rank(m.hard_negative_first_rank)}}</td></tr>`}}).join('')}}</tbody></table></div></div><h3>各阶段排行榜</h3>${{ORDER.map(s=>`<div class="stage-block"><h4>${{STAGES[s]}} · 本阶段排名 / 跨阶段排名轨迹 · ${{c.stage_data?.[s]?.candidates?.length||0}} 条</h4>${{(c.stage_data?.[s]?.candidates||[]).slice(0,DATA.config.default_visible).map(x=>candidateCard(x,s)).join('')||'<div class="muted">该阶段未执行或没有记录。</div>'}}</div>`).join('')}}<details><summary>required_facts / forbidden_claims（仅供生成评测参考，本报告不评分）</summary><pre>${{esc(JSON.stringify({{required_facts:c.required_facts,forbidden_claims:c.forbidden_claims}},null,2))}}</pre></details></div></details>`}}
// 重新定义总表渲染以显示 baseline 的对→错/错→对变化；保持过滤和排序逻辑一致。
function renderCaseTable(){{const q=$('search').value.trim().toLowerCase(), b=$('behavior').value, d=$('diagnostic').value, sort=$('sort').value, stage=DATA.config.default_stage; let rows=DATA.cases.filter(c=>{{const hay=[c.id,c.question,...(c.stage_data?.[stage]?.candidates||[]).map(x=>x.filename)].join(' ').toLowerCase(); return (!q||hay.includes(q))&&(b==='all'||c.expected_behavior===b)&&(d==='all'||(c.diagnostics||[]).some(x=>x.code===d))}}); rows.sort((a,b)=>sort==='id'?a.id.localeCompare(b.id):sort==='coverage'?((b.stage_summaries?.[stage]?.gold_coverage||0)-(a.stage_summaries?.[stage]?.gold_coverage||0)):sort==='rank'?((a.stage_summaries?.[stage]?.first_gold_rank||999)-(b.stage_summaries?.[stage]?.first_gold_rank||999)):priorityScore(b)-priorityScore(a)); const escChange=c=>{{const state=DATA.baseline?.case_changes?.[c.id]?.state||'—'; const color=state==='对→错'?'#b42318':state==='错→对'?'#087a42':'#667085'; return `<span style="color:${{color}};font-weight:700">${{esc(state)}}</span>`}}; $('caseTable').innerHTML=`<div class="filter-note">显示 ${{rows.length}} / ${{DATA.cases.length}} 题；点击行查看详情。</div><table><thead><tr><th>题号</th><th>问题</th><th>预期</th><th>默认阶段证据覆盖</th><th>首条金标</th><th>首个负例</th><th>诊断</th><th>变化</th></tr></thead><tbody>${{rows.map(c=>{{const m=c.stage_summaries?.[stage]||{{}}; return `<tr class="clickable" data-case="${{esc(c.id)}}"><td><b>${{esc(c.id)}}</b></td><td>${{esc(c.question.slice(0,70))}}</td><td>${{esc(c.expected_behavior)}}</td><td>${{m.gold_found??'—'}}/${{m.gold_total??'—'}}</td><td>${{rank(m.first_gold_rank)}}</td><td>${{rank(m.hard_negative_first_rank)}}</td><td>${{labels(c)}}</td><td>${{escChange(c)}}</td></tr>`}}).join('')}}</tbody></table>`; document.querySelectorAll('#caseTable [data-case]').forEach(e=>e.onclick=()=>openCase(e.dataset.case))}}
// 用更明确的标签、分数和高亮覆盖基础卡片渲染；保留基础详情结构。
const baseDetailHtml=detailHtml;
function candidateCard(c,stage){{const category=c.category||'unlabelled'; const text=(c.included_text??c.text??''); const kinds=s=>s.kinds||[]; const content=(c.highlight_segments||[]).map(s=>{{const k=kinds(s); const cls=k.includes('positive')&&k.includes('hard_negative')?'mixed':k.includes('positive')?(k.includes('partial')?'partial-gold':'gold'):k.includes('hard_negative')?(k.includes('partial')?'partial-negative':'negative'):''; return `<span class="hl-${{cls}}">${{esc(s.text)}}</span>`}}).join('')||esc(text); const history=(c.rank_history||[]).map(h=>`${{STAGES[h.stage]||h.stage}} #${{h.rank}}`).join(' → '); const labels={{gold:'金标证据',negative:'指定负例',mixed:'同块混合',unlabelled:'相似未标注',partial_gold:'部分金标',partial_negative:'部分负例'}}; const place=c.location||(c.page_number!=null?`第 ${{c.page_number}} 页`:''); return `<div class="candidate ${{category}}"><div class="candidate-head"><span class="rank">${{c.rank_display==='未入选'?'未入选':'#'+esc(c.rank_display)}}</span><b>${{esc(c.filename||'')}}</b><span class="pill">${{esc(place)}}</span><span class="badge ${{category}}">${{labels[category]||esc(category)}}</span><span class="score">关键词 ${{esc(c.lexical_score??'—')}} · 向量 ${{esc(c.vector_score??'—')}} · 融合 ${{esc(c.fused_score??'—')}} · 重排 ${{esc(c.rerank_score??'—')}} · 综合重排 ${{esc(c.rerank_score_combined??'—')}} · 直接回答支持 ${{esc(c.answer_support_score??'—')}} · 约束 ${{esc(c.constraint_score??'—')}}</span></div><div class="muted">状态：${{esc(c.status||'')}}；原因：${{esc(c.reason||c.constraint_decision?.reason||'')}}<br>排名轨迹：${{esc(history||'未进入后续阶段')}}</div><pre>${{content}}</pre></div>`}}
function enhancedDetailHtml(c){{let html=baseDetailHtml(c); const extras=ORDER.map(s=>{{const all=c.stage_data?.[s]?.candidates||[]; const rest=all.slice(DATA.config.default_visible); return rest.length?`<details class="stage-extra"><summary class="expand-stage">展开${{STAGES[s]||s}}剩余 ${{rest.length}} 条</summary>${{rest.map(x=>candidateCard(x,s)).join('')}}</details>`:''}}).join(''); return extras?html.replace('<details><summary>required_facts',extras+'<details><summary>required_facts'):html}}
function renderDetails(){{$('caseDetails').innerHTML=DATA.cases.map(enhancedDetailHtml).join('')}}
function openCase(id){{const e=$('case-'+id); if(e){{e.open=true;e.scrollIntoView({{behavior:'smooth',block:'start'}})}}}}
function renderExcluded(){{$('excluded').innerHTML=DATA.cases.filter(c=>!c.score_eligible).map(c=>`<div class="evidence miss"><b>${{esc(c.id)}}</b>：${{esc(c.excluded_reason||'未计分')}} · ${{esc(c.expected_behavior)}}</div>`).join('')||'<div class="muted">没有未计分题。</div>'; $('definitions').innerHTML='<h3>指标说明</h3>'+Object.entries(DATA.metric_definitions||{{}}).map(([k,v])=>`<p><b>${{esc(k)}}：</b>${{esc(v)}}</p>`).join('')+'<p class="muted">只对明确声明且完整覆盖的 Gold 计分；本报告不是完整语料的真实 Recall，也不是答案语义正确率。</p>'}}
function init(){{renderBanner();renderSummary();renderTiming();renderPriority();renderCharts();renderHeatmap();const codes=[...new Set(DATA.cases.flatMap(c=>(c.diagnostics||[]).map(d=>d.code)))]; $('diagnostic').innerHTML='<option value="all">全部诊断</option>'+codes.map(code=>`<option value="${{esc(code)}}">${{esc((DATA.cases.flatMap(c=>c.diagnostics||[]).find(d=>d.code===code)||{{}}).title||code)}}</option>`).join(''); renderCaseTable();renderDetails();renderExcluded();['search','behavior','diagnostic','sort'].forEach(id=>$(id).addEventListener('input',renderCaseTable));$('trendMetric').onchange=renderCharts;$('heatMetric').onchange=renderHeatmap;$('clear').onclick=()=>{{['search','behavior','diagnostic'].forEach(id=>$(id).value=id==='search'?'':'all');renderCaseTable()}}}}
init();
</script></body></html>'''
    # 保持单文件离线报告；导航源码在生成时直接写进 HTML。
    script_marker = "\ninit();\n</script>"
    if script_marker not in html_text:
        raise ValueError("报告模板缺少导航插入位置")
    html_text = html_text.replace(
        '<section class="panel"><h2>优先排查清单</h2>',
        '<section class="panel"><h2>查询路线与证据去向</h2>'
        '<div id="diagnosticOverview" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px"></div>'
        '</section><section class="panel"><h2>优先排查清单</h2>',
        1,
    )
    html_text = html_text.replace(
        "renderCharts();renderHeatmap();", "renderCharts();renderDiagnosticOverview();renderHeatmap();", 1
    )
    return html_text.replace(
        "</head>", f"<style>{NAVIGATION_CSS}{TIMING_CSS}</style></head>", 1
    ).replace(script_marker, f"\n{TIMING_JS}\n{OVERVIEW_JS}\n{NAVIGATION_JS}\ninit();\n</script>", 1)


def generate_report(
    cases_path: Path,
    results_path: Path,
    output_dir: Path,
    config: ReportConfig | None = None,
    *,
    baseline_path: Path | None = None,
) -> dict[str, Any]:
    config = config or ReportConfig()
    dataset = load_gold_dataset(cases_path, reviewed_only=False)
    metadata, results = _read_results(results_path)
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8")) if baseline_path else None
    detail = build_report_detail(dataset, metadata, results, config, baseline=baseline)
    write_report_files(detail, Path(output_dir).expanduser().resolve())
    if detail["validation"]["mismatch"] and not config.allow_snapshot_mismatch:
        raise SnapshotMismatchError(detail["validation"]["warning"])
    return detail


class SnapshotMismatchError(RuntimeError):
    """报告已写出，但索引版本不一致，命令行应返回非零状态。"""
