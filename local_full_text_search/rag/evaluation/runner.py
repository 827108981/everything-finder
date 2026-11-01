from __future__ import annotations

import json
import html
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from local_full_text_search.config.constants import APP_DATA_DIR
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.evaluation.evidence_matcher import GoldEvidenceMatch, load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import GoldCase, GoldDataset, load_gold_dataset
from local_full_text_search.rag.evaluation.metrics import (
    METRIC_DEFINITIONS,
    STAGE_ORDER,
    STAGE_LABELS,
    aggregate,
    build_rank_history,
    score_case,
)
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.lexical_index_store import LexicalIndexStore
from local_full_text_search.rag.models import RagAnswer, RagQuery, RagRetrievalResult
from local_full_text_search.rag.retrieval.lexical_analyzer import get_lexical_analyzer
from local_full_text_search.rag.retrieval.vector_units import VectorUnitPolicy
from local_full_text_search.rag.vector_index_store import VectorIndexStore
from local_full_text_search.rag.evaluation.reporting.service import (
    ReportConfig,
    build_report_detail,
    write_report_files,
)
from local_full_text_search.rag.evaluation.transitions import build_candidate_transitions


class RagEvalError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RagEvalConfig:
    dataset_path: Path
    mode: str = "retrieval-only"
    case_ids: tuple[str, ...] = ()
    output_root: Path | None = None
    # 仅用于在资料内容未变、但 AI 索引已经重建或更换模型时做诊断。
    # 默认关闭，避免把不同语料快照的结果误当成正式指标。
    allow_snapshot_mismatch: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"retrieval-only", "full"}:
            raise ValueError("评测模式只能为 retrieval-only 或 full")


@dataclass(frozen=True, slots=True)
class PreparedEval:
    dataset: GoldDataset
    bundle_id: str
    cases: tuple[GoldCase, ...]
    evidence: dict[str, tuple[GoldEvidenceMatch, ...]]
    coverage: dict[str, Any]


class RagEvalRunner:
    """无 Qt 依赖的评测循环；UI 和测试都调用此类。"""

    def __init__(
        self,
        generation_base_path: Path,
        settings,
        token: CancelToken,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.generation_base_path = Path(generation_base_path).resolve()
        self.settings = settings
        self.token = token
        self.progress_callback = progress_callback
        self.last_report: dict[str, Any] | None = None
        self.redaction_values: tuple[str, ...] = ()

    def prepare(self, config: RagEvalConfig) -> PreparedEval:
        # 批量运行只执行人工审核通过的题；待审核/争议题仍保留在数据集对象，
        # 由诊断报告列在未计分区域，避免它们悄悄消失。
        dataset = load_gold_dataset(config.dataset_path, reviewed_only=False)
        requested_ids = set(config.case_ids)
        selected = tuple(
            case for case in dataset.cases
            if case.review_status == "reviewed" and (not requested_ids or case.case_id in requested_ids)
        )
        if config.case_ids:
            missing = sorted(set(config.case_ids) - {case.case_id for case in dataset.cases if case.review_status == "reviewed"})
            if missing:
                raise RagEvalError("找不到指定题号：" + ", ".join(missing))
        if not selected:
            raise RagEvalError("测试集中没有 review_status=reviewed 的可评测题目")
        rag_store = RagIndexStore(self.generation_base_path.parent / "rag")
        bundle = rag_store.active()
        snapshots = {case.corpus_snapshot for case in selected}
        if "" in snapshots:
            raise RagEvalError("测试集缺少 corpus_snapshot，无法确认评测所用 AI 索引版本")
        mismatched = sorted(snapshot for snapshot in snapshots if snapshot and snapshot != bundle.bundle_id)
        snapshot_mismatch = bool(mismatched)
        if snapshot_mismatch and not config.allow_snapshot_mismatch:
            raise RagEvalError(
                "测试集 corpus_snapshot 与当前已发布 AI 索引不一致。\n"
                f"测试集：{', '.join(mismatched)}\n当前：{bundle.bundle_id}\n"
                "请切换到对应索引、更新测试集版本，或明确勾选“允许快照不一致（仅诊断）”后再评测。"
            )
        connection = rag_store.connect_read_only(bundle.metadata_path)
        try:
            chunk_rows = load_chunk_rows(connection)
        finally:
            connection.close()
        evidence = {
            case.case_id: tuple(map_case_evidence(case, chunk_rows))
            for case in selected
        }
        positive_total = sum(1 for case in selected for _ in case.gold_evidence)
        positive_found = sum(
            1 for case in selected for item in evidence[case.case_id]
            if item.kind == "positive" and item.chunk_ids
        )
        review_count = sum(
            1 for matches in evidence.values() for item in matches
            if item.confidence in {"review", "unmatched"}
        )
        coverage = {
            "case_count": len(selected),
            "dataset_case_count": len(dataset.cases),
            "reviewed_case_count": sum(case.review_status == "reviewed" for case in selected),
            "positive_evidence_total": positive_total,
            "positive_evidence_located": positive_found,
            "positive_index_coverage": positive_found / positive_total if positive_total else None,
            "evidence_needing_review": review_count,
            "refusal_case_count": sum(case.expected_behavior == "refuse" for case in selected),
            "behavior_counts": {
                behavior: sum(case.expected_behavior == behavior for case in selected)
                for behavior in ("answer", "partial_answer", "clarify", "refuse")
            },
            "snapshot_mismatch": snapshot_mismatch,
            "snapshot_override": bool(config.allow_snapshot_mismatch and snapshot_mismatch),
            "expected_bundle_ids": sorted(snapshots),
            "current_bundle_id": bundle.bundle_id,
        }
        return PreparedEval(dataset, bundle.bundle_id, selected, evidence, coverage)

    def run(
        self,
        config: RagEvalConfig,
        prepared: PreparedEval,
        service_factory: Callable[[Callable[[dict[str, Any]], None]], Any],
        judge_factory: Callable[[Callable[[dict[str, Any]], None]], Any] | None = None,
        preparation_duration_ms: int | None = None,
    ) -> dict[str, Any]:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        output_root = Path(config.output_root or (APP_DATA_DIR / "rag_eval_runs")).expanduser().resolve()
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        started = datetime.now(timezone.utc).isoformat()
        lexical_index: dict[str, Any] = {}
        if str(getattr(self.settings, "rag_lexical_backend", "legacy")) == "word_v2":
            analyzer = get_lexical_analyzer()
            attached = LexicalIndexStore(self.generation_base_path.parent / "rag").active(
                prepared.bundle_id, analyzer.fingerprint,
            )
            lexical_index = {
                "source_bundle_id": attached.source_bundle_id,
                "index_fingerprint": attached.index_fingerprint,
                "analyzer_fingerprint": attached.analyzer_fingerprint,
                "chunk_count": attached.chunk_count,
                "database_path": str(attached.database_path),
            }
        vector_index: dict[str, Any] = {}
        if str(getattr(self.settings, "rag_vector_backend", "legacy")) == "unit_v2":
            policy = VectorUnitPolicy.from_settings(self.settings)
            vector_rag_store = RagIndexStore(self.generation_base_path.parent / "rag")
            active_bundle = vector_rag_store.active()
            vector_store = VectorIndexStore(vector_rag_store.root)
            if getattr(active_bundle, "vector_unit_directory", ""):
                attached_vector = vector_store.attached(
                    prepared.bundle_id,
                    active_bundle.vector_unit_directory,
                    embedding_repo_id=str(getattr(self.settings, "rag_embedding_repo_id", "")),
                    embedding_revision=str(getattr(self.settings, "rag_embedding_revision", "")),
                    embedding_dimension=active_bundle.embedding_dimension,
                    policy_fingerprint=policy.fingerprint,
                    index_fingerprint=active_bundle.vector_unit_index_fingerprint,
                )
            else:
                attached_vector = vector_store.active(
                    prepared.bundle_id,
                    embedding_repo_id=str(getattr(self.settings, "rag_embedding_repo_id", "")),
                    embedding_revision=str(getattr(self.settings, "rag_embedding_revision", "")),
                    embedding_dimension=active_bundle.embedding_dimension,
                    policy_fingerprint=policy.fingerprint,
                )
            vector_index = {
                "source_bundle_id": attached_vector.source_bundle_id,
                "index_fingerprint": attached_vector.index_fingerprint,
                "policy_fingerprint": attached_vector.policy_fingerprint,
                "parent_chunk_count": attached_vector.parent_chunk_count,
                "unit_count": attached_vector.unit_count,
                "database_path": str(attached_vector.database_path),
            }
        metadata = {
            "run_id": run_id,
            "started_at": started,
            "preparation_duration_ms": preparation_duration_ms,
            "mode": config.mode,
            "dataset_path": str(prepared.dataset.path),
            "dataset_sha256": prepared.dataset.sha256,
            "ai_bundle_id": prepared.bundle_id,
            "settings": {
                "embedding_model": getattr(self.settings, "rag_embedding_repo_id", ""),
                "embedding_revision": getattr(self.settings, "rag_embedding_revision", ""),
                "reranker_model": getattr(self.settings, "rag_reranker_repo_id", ""),
                "reranker_revision": getattr(self.settings, "rag_reranker_revision", ""),
                "rerank_weights": {
                    "cross_encoder": float(getattr(self.settings, "rag_rerank_cross_encoder_weight", 0.65)),
                    "retrieval_prior": float(getattr(self.settings, "rag_rerank_retrieval_prior_weight", 0.05)),
                    "answer_support": float(getattr(self.settings, "rag_rerank_answer_support_weight", 0.30)),
                    "max_rank_drop": int(getattr(self.settings, "rag_rerank_max_rank_drop", 3)),
                },
                "llm_model": getattr(self.settings, "rag_api_model", "") if config.mode == "full" else "未调用",
                "api_style": getattr(self.settings, "rag_api_style", "") if config.mode == "full" else "未调用",
                "answer_review_enabled": bool(getattr(self.settings, "rag_answer_review_enabled", False)) if config.mode == "full" else False,
                "lexical_multi_route_enabled": bool(getattr(self.settings, "rag_lexical_multi_route_enabled", False)),
                "lexical_backend": str(getattr(self.settings, "rag_lexical_backend", "legacy")),
                "lexical_title_weight": float(getattr(self.settings, "rag_lexical_title_weight", 0.2)),
                "lexical_complete_match_enabled": bool(getattr(
                    self.settings, "rag_lexical_complete_match_enabled", True
                )),
                "lexical_k1": float(getattr(self.settings, "rag_lexical_k1", 1.2)),
                "lexical_b": float(getattr(self.settings, "rag_lexical_b", 0.75)),
                "lexical_index": lexical_index,
                "vector_focused_query_enabled": bool(getattr(self.settings, "rag_vector_focused_query_enabled", False)),
                "vector_backend": str(getattr(self.settings, "rag_vector_backend", "legacy")),
                "vector_index": vector_index,
                "vector_unit_policy": {
                    "target_tokens": int(getattr(self.settings, "rag_vector_unit_target_tokens", 128)),
                    "max_tokens": int(getattr(self.settings, "rag_vector_unit_max_tokens", 192)),
                    "overlap_tokens": int(getattr(self.settings, "rag_vector_unit_overlap_tokens", 32)),
                    "include_source_context": bool(getattr(
                        self.settings, "rag_vector_unit_include_source_context", True
                    )),
                    "oversample_factor": int(getattr(
                        self.settings, "rag_vector_unit_oversample_factor", 8
                    )),
                },
                "route_rrf_k": 60,
                "limits": {
                    "lexical_top_k": int(self.settings.rag_lexical_top_k),
                    "vector_top_k": int(self.settings.rag_vector_top_k),
                    "fusion_top_k": int(self.settings.rag_fusion_top_k),
                    "reranker_top_k": int(self.settings.rag_reranker_top_k),
                    "final_top_k": int(self.settings.rag_final_top_k),
                },
            },
            "preflight": prepared.coverage,
        }
        self._write_json(run_dir / "run.json", metadata)
        results: list[dict[str, Any]] = []
        self.last_report = None
        cases_path = run_dir / "cases.jsonl"
        self._emit({"type": "run_started", "run_id": run_id, "run_dir": str(run_dir), "total": len(prepared.cases), "preflight": prepared.coverage})
        try:
            for index, case in enumerate(prepared.cases, 1):
                self.token.throw_if_cancelled()
                case_started = time.monotonic()
                stages: dict[str, list[dict[str, Any]]] = {}
                events: list[dict[str, Any]] = []
                last_event_at = case_started

                def on_progress(payload: dict[str, Any]) -> None:
                    nonlocal last_event_at
                    now = time.monotonic()
                    for field, stage in (
                        ("lexical_candidates", "lexical"),
                        ("vector_candidates", "vector"),
                        ("fused_candidates", "fusion"),
                        ("deduplicated_candidates", "deduplicated"),
                        ("reranked_candidates", "rerank"),
                        ("constraint_candidates", "constraints"),
                        ("evidence_candidates", "evidence"),
                    ):
                        candidates = payload.get(field)
                        if isinstance(candidates, list):
                            stages[stage] = [dict(item) for item in candidates if isinstance(item, dict)]
                    event = {
                        key: value for key, value in payload.items()
                        if key not in {
                            "lexical_candidates", "vector_candidates", "fused_candidates",
                            "deduplicated_candidates", "reranked_candidates",
                            "constraint_candidates", "evidence_candidates",
                        }
                    }
                    event["elapsed_ms"] = int((now - case_started) * 1000)
                    event["since_previous_event_ms"] = int((now - last_event_at) * 1000)
                    last_event_at = now
                    events.append(event)
                    self._emit({"type": "stage", "index": index, "total": len(prepared.cases), "case_id": case.case_id, "payload": payload})

                case_query = RagQuery(
                    text=case.question,
                    include_ocr=True,
                    min_ocr_confidence=float(self.settings.ocr_min_confidence),
                    lexical_top_k=self.settings.rag_lexical_top_k,
                    vector_top_k=self.settings.rag_vector_top_k,
                    fusion_top_k=self.settings.rag_fusion_top_k,
                    reranker_top_k=self.settings.rag_reranker_top_k,
                    final_top_k=self.settings.rag_final_top_k,
                )
                try:
                    service = service_factory(on_progress)
                    trace_id = f"eval-{run_id}-{case.case_id}"
                    if config.mode == "retrieval-only":
                        retrieval = service.retrieve(case_query, self.token, trace_id=trace_id)
                        answer_record = _retrieval_record(retrieval)
                    else:
                        answer = service.answer(case_query, self.token, trace_id=trace_id)
                        answer_record = _answer_record(answer)
                    gold_judgement = None
                    if config.mode == "full" and judge_factory is not None and str(answer_record.get("text") or "").strip():
                        on_progress({"stage": "gold_judging", "message": "正在将最终答案与人工金标逐项比较"})
                        try:
                            gold_judgement = judge_factory(on_progress).evaluate(case, answer_record, self.token)
                        except Exception as judge_exc:
                            if self.token.cancelled:
                                raise
                            gold_judgement = {"verdict": "unavailable", "method": "llm_gold_judge",
                                              "reason": self._redact(str(judge_exc))}
                        on_progress({"stage": "gold_judged", "verdict": gold_judgement["verdict"],
                                     "message": "金标答案裁判完成"})
                    elif config.mode == "full" and judge_factory is not None:
                        gold_judgement = {"verdict": "unavailable", "method": "llm_gold_judge",
                                          "reason": "本题没有可核查的生成答案"}
                    # 上下文事件按代表正文记录顺序；retrieval.sources 是扁平化来源列表，
                    # 其中同一正文的 PDF/PPTX 来源不能各占一个上下文名次。
                    context_event = next(
                        (item for item in reversed(events) if item.get("context_entries") or item.get("context_sources")),
                        None,
                    )
                    if context_event:
                        entries = context_event.get("context_entries") or context_event.get("context_sources") or []
                        selected = {
                            str(item.get("chunk_id") or ""): item
                            for item in stages.get("evidence", []) if item.get("selected") is True
                        }
                        context_rows: list[dict[str, Any]] = []
                        for entry in entries:
                            if not isinstance(entry, dict):
                                continue
                            chunk_id = str(entry.get("chunk_id") or "")
                            if chunk_id not in selected:
                                continue
                            context_rows.append({
                                **selected[chunk_id],
                                "included_text": entry.get("included_text"),
                                "truncated": bool(entry.get("truncated")),
                                "context_reason": entry.get("reason", ""),
                                "context_rank": len(context_rows) + 1,
                            })
                        stages["context"] = context_rows
                    scored = score_case(case, stages, list(prepared.evidence[case.case_id]), answer=answer_record)
                    transitions = build_candidate_transitions(
                        stages, events, reranker_top_k=int(self.settings.rag_reranker_top_k),
                    )
                    case_result = {
                        **scored,
                        "case_index": index,
                        "duration_ms": int((time.monotonic() - case_started) * 1000),
                        "api_call_count": _api_attempt_count(events),
                        "gold_judgement": gold_judgement,
                        "candidate_stages": stages,
                        "candidate_transitions": transitions,
                        "events": events,
                        "required_facts": list(case.required_facts),
                        "forbidden_claims": list(case.forbidden_claims),
                    }
                except Exception as exc:
                    if self.token.cancelled:
                        raise
                    case_result = {
                        "id": case.case_id,
                        "group_id": case.group_id,
                        "question": case.question,
                        "expected_behavior": case.expected_behavior,
                        "actual_behavior": "error",
                        "behavior_match": None,
                        "error": {"type": type(exc).__name__, "message": self._redact(str(exc))},
                        "failed_stage": str(events[-1].get("stage") if events else "model_loading"),
                        "duration_ms": int((time.monotonic() - case_started) * 1000),
                        "api_call_count": _api_attempt_count(events),
                        "candidate_stages": stages,
                        "events": events,
                    }
                results.append(case_result)
                with cases_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(case_result, ensure_ascii=False, default=str) + "\n")
                    stream.flush()
                self._emit({"type": "case_finished", "index": index, "total": len(prepared.cases), "result": case_result})
                # LLM/API 失败时不对剩余题目重复发送相同请求。
                if config.mode == "full" and case_result.get("error"):
                    break
        except Exception:
            metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
            metadata["state"] = "cancelled" if self.token.cancelled else "failed"
            metadata["completed_case_count"] = len(results)
            self._write_json(run_dir / "run.json", metadata)
            summary = aggregate(results)
            self._write_json(run_dir / "summary.json", summary)
            self._write_summary_markdown(run_dir / "summary.md", metadata, summary, results)
            self._write_report_html(run_dir / "report.html", metadata, summary, results)
            self._write_diagnostic_report(run_dir, prepared.dataset, metadata, results)
            self.last_report = {
                "run_id": run_id, "run_dir": str(run_dir), "summary": summary,
                "results": results, "state": metadata["state"],
            }
            self._emit({"type": "run_partial", **self.last_report})
            raise
        summary = aggregate(results)
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        metadata["state"] = "completed" if len(results) == len(prepared.cases) else "partial"
        metadata["completed_case_count"] = len(results)
        self._write_json(run_dir / "run.json", metadata)
        self._write_json(run_dir / "summary.json", summary)
        self._write_summary_markdown(run_dir / "summary.md", metadata, summary, results)
        self._write_report_html(run_dir / "report.html", metadata, summary, results)
        self._write_diagnostic_report(run_dir, prepared.dataset, metadata, results)
        report = {"run_id": run_id, "run_dir": str(run_dir), "summary": summary, "results": results, "state": metadata["state"]}
        self._emit({"type": "run_finished", **report})
        return report

    def _emit(self, payload: dict[str, Any]) -> None:
        if self.progress_callback:
            self.progress_callback(payload)

    def _redact(self, value: str) -> str:
        for secret in self.redaction_values:
            if secret:
                value = value.replace(secret, "[已隐藏]")
        return value

    @staticmethod
    def _write_diagnostic_report(
        run_dir: Path,
        dataset: GoldDataset,
        metadata: dict[str, Any],
        results: list[dict[str, Any]],
    ) -> None:
        """生成新版诊断产物；报告失败不能覆盖已经落盘的原始评测结果。"""

        try:
            detail = build_report_detail(
                dataset,
                metadata,
                results,
                ReportConfig(
                    allow_snapshot_mismatch=bool(
                        metadata.get("preflight", {}).get("snapshot_override", False)
                    )
                ),
            )
            write_report_files(detail, run_dir)
        except Exception:
            # 评测结果已经由上层原子写入。诊断报告属于派生物，失败时保留
            # 旧版 report.html 和 cases.jsonl，供命令行重新生成。
            return

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _write_summary_markdown(path: Path, metadata: dict[str, Any], summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
        lines = [
            "# RAG 批量评测报告", "", f"- 运行编号：{metadata['run_id']}",
            f"- 模式：{metadata['mode']}", f"- AI 索引：{metadata['ai_bundle_id']}",
            f"- 完成题数：{summary['completed_case_count']}/{summary['case_count']}",
            f"- 行为匹配率：{_format_metric(summary.get('behavior_accuracy'))}", "",
            f"- API 请求数：{summary.get('api_call_count', 0)}", "",
            "## 各检索阶段", "", "| 阶段 | Gold 命中题数 | 命中率 | 证据召回率 | MRR | 首位排名均值 | 反例候选数 |", "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for stage, metrics in summary["stages"].items():
            lines.append(
                f"| {STAGE_LABELS.get(stage, stage)} | {metrics['gold_hit_cases']}/{metrics['cases_with_gold']} | "
                f"{_format_metric(metrics['gold_hit_rate'])} | {_format_metric(metrics['mean_evidence_recall'])} | "
                f"{_format_metric(metrics['mean_reciprocal_rank'])} | {metrics['mean_first_gold_rank'] or '—'} | "
                f"{metrics['hard_negative_candidate_count']} |"
            )
        lines.extend(["", "## 指标口径", ""])
        lines.extend(f"- {definition}" for definition in METRIC_DEFINITIONS.values())
        lines.extend(["", "## 逐题首次命中", "", "| 题号 | 预期 | 实际 | 首次发现 Gold 的阶段 | 禁止结论命中 |", "|---|---|---|---|---:|"])
        for result in results:
            first = sorted(set(result.get("first_gold_hit", {}).values()), key=lambda item: STAGE_ORDER.index(item))
            lines.append(
                f"| {result.get('id','')} | {result.get('expected_behavior','')} | {result.get('actual_behavior','')} | "
                f"{first[0] if first else '未命中/无正证据'} | {len(result.get('forbidden_claims_found', []))} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _write_report_html(path: Path, metadata: dict[str, Any], summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
        esc = lambda value: html.escape(str(value or ""))
        lines = [
            "<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">",
            "<title>RAG 批量评测报告</title>",
            "<style>body{font:15px system-ui,sans-serif;margin:28px;color:#172033}"
            "table{border-collapse:collapse;width:100%;margin:12px 0 24px}"
            ".table-wrap{overflow-x:auto;width:100%}.table-wrap table{min-width:1450px}"
            "th,td{border:1px solid #ccd3df;padding:7px;text-align:left;vertical-align:top}"
            "th{background:#f1f4f8}details{margin:12px 0;padding:10px;border:1px solid #d8deea;border-radius:6px}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f8fa;padding:10px}"
            ".ok{color:#087a42}.bad{color:#b42318}.muted{color:#667085}</style><body>",
            "<h1>RAG 批量评测报告</h1>",
            f"<p>运行：{esc(metadata.get('run_id'))}　模式：{esc(metadata.get('mode'))}　AI 索引：{esc(metadata.get('ai_bundle_id'))}</p>",
            f"<p>完成题数：{summary.get('completed_case_count', 0)}/{summary.get('case_count', 0)}　行为匹配率：{esc(_format_metric(summary.get('behavior_accuracy')))}</p>",
            f"<p>API 请求数：{summary.get('api_call_count', 0)}</p>",
            "<h2>检索阶段汇总</h2><table><tr><th>阶段</th><th>Gold 命中题数</th><th>命中率</th><th>平均证据召回率</th><th>MRR</th><th>首位排名均值</th><th>指定反例候选数</th></tr>",
        ]
        for stage, values in summary.get("stages", {}).items():
            lines.append(
                f"<tr><td>{esc(STAGE_LABELS.get(stage, stage))}</td><td>{values.get('gold_hit_cases', 0)}/{values.get('cases_with_gold', 0)}</td>"
                f"<td>{esc(_format_metric(values.get('gold_hit_rate')))}</td>"
                f"<td>{esc(_format_metric(values.get('mean_evidence_recall')))}</td>"
                f"<td>{esc(_format_metric(values.get('mean_reciprocal_rank')))}</td>"
                f"<td>{esc(values.get('mean_first_gold_rank') or '—')}</td>"
                f"<td>{values.get('hard_negative_candidate_count', 0)}</td></tr>"
            )
        lines.append("</table><details><summary>指标名称与口径说明</summary><ul>")
        lines.extend(f"<li>{esc(definition)}</li>" for definition in METRIC_DEFINITIONS.values())
        lines.append("</ul></details><h2>逐题记录</h2>")
        for result in results:
            first = sorted(
                set(result.get("first_gold_hit", {}).values()),
                key=lambda item: STAGE_ORDER.index(item) if item in STAGE_ORDER else len(STAGE_ORDER),
            )
            answer = result.get("answer") or {}
            candidate_stages = result.get("candidate_stages", {})
            rank_history = build_rank_history(candidate_stages)
            positive_ids = {
                str(chunk_id)
                for item in result.get("gold_mappings", [])
                if item.get("kind") == "positive"
                for chunk_id in item.get("chunk_ids", [])
            }
            negative_ids = {
                str(chunk_id)
                for item in result.get("gold_mappings", [])
                if item.get("kind") == "hard_negative"
                for chunk_id in item.get("chunk_ids", [])
            }
            match = result.get("behavior_match")
            state_class = "muted" if match is None else "ok" if match else "bad"
            lines.append(
                f"<details><summary><b>{esc(result.get('id'))}</b>　预期：{esc(result.get('expected_behavior'))}　"
                f"实际：<span class=\"{state_class}\">{esc(result.get('actual_behavior'))}</span>　"
                f"首次命中：{esc(first[0] if first else '未命中/无正证据')}</summary>"
                f"<p>{esc(result.get('question'))}</p><h3>答案</h3><pre>{esc(answer.get('text') or '未生成')}</pre>"
            )
            for stage in STAGE_ORDER:
                candidates = candidate_stages.get(stage, [])
                if not candidates:
                    continue
                lines.append(f"<h3>{esc(STAGE_LABELS.get(stage, stage))}　候选 {len(candidates)} 条</h3><div class=\"table-wrap\"><table><tr><th>本阶段排名</th><th>Gold/反例标注</th><th>文件/位置</th><th>跨阶段排名轨迹</th><th>关键词名次</th><th>向量名次</th><th>向量分数</th><th>RRF 分数</th><th>重排分数</th><th>处理原因与正文</th></tr>")
                for index, candidate in enumerate(candidates, 1):
                    chunk_id = str(candidate.get("chunk_id") or "")
                    labels = []
                    if chunk_id in positive_ids:
                        labels.append("正确证据")
                    if chunk_id in negative_ids:
                        labels.append("指定反例")
                    label = " + ".join(labels) if labels else "未判定"
                    if stage == "evidence":
                        stage_rank = str(candidate.get("final_rank") or "未入选")
                    else:
                        stage_rank = str(index)
                    trace = " → ".join(
                        f"{STAGE_LABELS.get(stage_name, stage_name)} #{rank}"
                        for stage_name, rank in rank_history.get(chunk_id, [])
                    ) or "未进入后续阶段"
                    lines.append(
                        f"<tr><td>{esc(stage_rank)}</td><td>{esc(label)}</td>"
                        f"<td>{esc(candidate.get('filename'))}<br>{esc(candidate.get('location'))}</td>"
                        f"<td>{esc(trace)}</td><td>{esc(candidate.get('lexical_rank'))}</td>"
                        f"<td>{esc(candidate.get('vector_rank'))}</td><td>{esc(candidate.get('vector_score'))}</td>"
                        f"<td>{esc(candidate.get('fused_score'))}</td><td>{esc(candidate.get('rerank_score'))}</td>"
                        f"<td>{esc(candidate.get('reason') or candidate.get('constraint_decision', {}).get('reason'))}"
                        f"<details><summary>查看正文</summary><pre>{esc(candidate.get('text'))}</pre></details></td></tr>"
                    )
                lines.append("</table></div>")
            lines.append("</details>")
        lines.append("<p class=\"muted\">未标注的候选仍为未判定；事实文字命中仅供定位，不代表语义正确。</p></body></html>")
        path.write_text("\n".join(lines), encoding="utf-8")


def _api_attempt_count(events: list[dict[str, Any]]) -> int:
    """包含失败题、自动重试及独立金标裁判的真实请求尝试次数。"""
    return sum(str(event.get("stage") or "") in {
        "generating", "generation_retry", "reviewing", "review_retry",
        "gold_judging", "gold_judge_retry",
    } for event in events)


def _answer_record(answer: RagAnswer) -> dict[str, Any]:
    return {
        "mode": "full",
        "status": str(getattr(answer.status, "value", answer.status)),
        "text": answer.answer_text,
        "insufficient_evidence": answer.insufficient_evidence,
        "trace_id": answer.trace_id,
        "provider": answer.provider_name,
        "model": answer.model_name,
        "source_ids": list(answer.source_ids),
        "sources": [
            {"source_id": answer.source_ids[index] if index < len(answer.source_ids) else f"S{index + 1}",
             "chunk_id": source.stable_chunk_id, "filename": source.filename,
             "location": source.location_text, "file_path": source.file_path}
            for index, source in enumerate(answer.sources)
        ],
        "warnings": list(answer.warnings),
        "review_verdict": answer.audit.review.verdict if answer.audit and answer.audit.review else "",
        "review_claims": [
            {"claim": claim.claim, "citations": list(claim.citations), "status": claim.status, "reason": claim.reason}
            for claim in (answer.audit.review.claims if answer.audit and answer.audit.review else ())
        ],
    }


def _retrieval_record(result: RagRetrievalResult) -> dict[str, Any]:
    return {
        "mode": "retrieval-only",
        "status": str(getattr(result.status, "value", result.status)),
        "text": "",
        "insufficient_evidence": result.insufficient_evidence,
        "trace_id": result.trace_id,
        "sources": [
            {"chunk_id": source.stable_chunk_id, "filename": source.filename,
             "location": source.location_text, "file_path": source.file_path}
            for source in result.sources
        ],
        "reason": result.reason,
    }


def _format_metric(value: object) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.1f}%"
