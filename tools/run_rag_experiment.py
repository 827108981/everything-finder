"""按 JSON 配置运行本机 RAG 评测；配置不得保存 API Key。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR, SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.runner import RagEvalConfig, RagEvalRunner
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore


_SETTING_KEYS = {
    "rag_lexical_multi_route_enabled", "rag_vector_focused_query_enabled",
    "rag_lexical_backend", "rag_lexical_title_weight",
    "rag_lexical_complete_match_enabled", "rag_lexical_k1", "rag_lexical_b",
    "rag_vector_backend", "rag_vector_unit_target_tokens",
    "rag_vector_unit_max_tokens", "rag_vector_unit_overlap_tokens",
    "rag_vector_unit_include_source_context", "rag_vector_unit_oversample_factor",
    "rag_rerank_cross_encoder_weight", "rag_rerank_retrieval_prior_weight",
    "rag_rerank_answer_support_weight",
    "rag_rerank_max_rank_drop",
    "rag_answer_review_enabled", "rag_lexical_top_k", "rag_vector_top_k",
    "rag_fusion_top_k", "rag_reranker_top_k", "rag_final_top_k",
    "rag_context_max_chars",
}
_CONFIG_KEYS = _SETTING_KEYS | {"generation_base_path", "expected_bundle_id"}


def load_experiment_config(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("实验配置须为 JSON 对象")
    unknown = set(raw) - _CONFIG_KEYS
    if unknown:
        raise ValueError("实验配置包含不支持的字段：" + ", ".join(sorted(unknown)))
    if not raw.get("generation_base_path") or not raw.get("expected_bundle_id"):
        raise ValueError("实验配置须指定 generation_base_path 和 expected_bundle_id")
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--mode", choices=("retrieval-only", "full"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="*", default=[])
    parser.add_argument(
        "--allow-snapshot-mismatch", action="store_true",
        help="仅用于诊断：允许测试集快照与当前 AI bundle 不一致，并在报告中标记",
    )
    args = parser.parse_args()

    started = time.monotonic()
    config_bytes = args.experiment_config.read_bytes()
    experiment = load_experiment_config(args.experiment_config)
    index_db = Path(str(experiment["generation_base_path"])).expanduser().resolve()
    settings = AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
    for key in _SETTING_KEYS & experiment.keys():
        setattr(settings, key, experiment[key])
    token = CancelToken()
    runner = RagEvalRunner(index_db, settings, token)
    eval_config = RagEvalConfig(
        dataset_path=args.cases, mode=args.mode, output_root=args.out,
        case_ids=tuple(args.case_ids),
        allow_snapshot_mismatch=args.allow_snapshot_mismatch,
    )
    prepared = runner.prepare(eval_config)
    if prepared.bundle_id != experiment["expected_bundle_id"]:
        raise ValueError("实际 AI 索引 bundle_id 与实验配置不一致，已停止运行")

    manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
    embedding_path = manager.resolve_local_path(ModelSpec(
        settings.rag_embedding_repo_id, settings.rag_embedding_revision, "embedding",
    ))
    reranker_path = manager.resolve_local_path(ModelSpec(
        settings.rag_reranker_repo_id, settings.rag_reranker_revision, "reranker",
    ))
    embedding = LocalBgeEmbeddingProvider(
        embedding_path, repo_id=settings.rag_embedding_repo_id,
        revision=settings.rag_embedding_revision, batch_size=settings.rag_embedding_batch_size,
    )
    reranker = LocalBgeReranker(
        reranker_path, repo_id=settings.rag_reranker_repo_id,
        revision=settings.rag_reranker_revision, batch_size=settings.rag_reranker_batch_size,
        max_length=settings.rag_reranker_max_length,
        cross_encoder_weight=getattr(settings, "rag_rerank_cross_encoder_weight", 0.65),
        retrieval_prior_weight=getattr(settings, "rag_rerank_retrieval_prior_weight", 0.05),
        answer_support_weight=getattr(settings, "rag_rerank_answer_support_weight", 0.30),
        max_rank_drop=getattr(settings, "rag_rerank_max_rank_drop", 3),
    )
    secrets = WindowsCredentialSecretStore()
    api_key = ""
    if args.mode == "full":
        if not str(settings.rag_api_model or "").strip():
            raise ValueError("设置页尚未配置 LLM 模型")
        api_key = secrets.retrieve(settings.rag_api_secret_ref) or ""
        if not api_key:
            raise ValueError("设置页没有已保存的 API Key")
        runner.redaction_values = (api_key,)

    def service_factory(callback):
        llm_stage = ["generation"]

        def record(event):
            if event.get("stage") == "assessing_clarification":
                llm_stage[0] = "clarification"
            elif event.get("stage") == "generating":
                llm_stage[0] = "generation"
            elif event.get("stage") == "reviewing":
                llm_stage[0] = "review"
            callback(event)

        gateway = OpenAICompatibleGateway(
            max_transient_retries=3 if args.mode == "full" else 0,
            retry_callback=lambda event: record({**event, "stage": llm_stage[0] + "_retry"}),
        )
        return AnswerService(
            IndexGenerationStore(index_db), RagIndexStore(index_db.parent / "rag"),
            embedding, reranker, gateway, secrets, settings,
            progress_callback=record,
        )

    def judge_factory(callback):
        gateway = OpenAICompatibleGateway(
            max_transient_retries=3,
            retry_callback=lambda event: callback({**event, "stage": "gold_judge_retry"}),
        )
        return GoldAnswerJudge(gateway, settings, api_key)

    result = runner.run(
        eval_config, prepared, service_factory,
        judge_factory=judge_factory if args.mode == "full" else None,
        preparation_duration_ms=int((time.monotonic() - started) * 1000),
    )
    run_dir = Path(str(result["run_dir"]))
    manifest = {
        "experiment_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "experiment_config": experiment,
        "mode": args.mode,
        "bundle_id": prepared.bundle_id,
        "case_ids": [case.case_id for case in prepared.cases],
        "completed_case_count": len(result.get("results") or []),
        "failed_case_count": sum(bool(row.get("error")) for row in result.get("results") or []),
        "state": result.get("state"),
        "api_key_in_config": False,
    }
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(run_dir), "state": result.get("state"),
                      "completed_case_count": manifest["completed_case_count"],
                      "failed_case_count": manifest["failed_case_count"]}, ensure_ascii=True))
    return 0 if result.get("state") == "completed" and not manifest["failed_case_count"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # 禁止打印 SDK 异常对象中的完整连接请求、凭据或资料正文。
        print(f"实验未完成：{type(exc).__name__}", file=sys.stderr)
        raise SystemExit(1)
