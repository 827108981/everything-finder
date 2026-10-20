"""从已经保存的 RAG 批量评测结果生成离线诊断报告。

这个入口只读取 JSON/JSONL，不加载 Embedding、Reranker 或 LLM。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from local_full_text_search.rag.evaluation.reporting.service import (
    ReportConfig,
    SnapshotMismatchError,
    generate_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 RAG 批量检索离线诊断报告")
    parser.add_argument("--cases", required=True, type=Path, help="Gold JSONL 或审核版 JSON 文件")
    parser.add_argument("--results", required=True, type=Path, help="运行目录或 cases.jsonl")
    parser.add_argument("--out", required=True, type=Path, help="报告输出目录")
    parser.add_argument("--topk", type=int, default=50, help="报告分析的最大阶段深度")
    parser.add_argument("--default-visible", type=int, default=10, help="单题默认显示候选数")
    parser.add_argument("--default-stage", choices=("lexical", "vector", "fusion", "deduplicated", "rerank", "constraints", "evidence", "context"), default="context")
    parser.add_argument("--page-tolerance", type=int, default=1)
    parser.add_argument("--high-match-threshold", type=float, default=0.60)
    parser.add_argument("--partial-match-threshold", type=float, default=0.30)
    parser.add_argument("--count-partial-matches", action="store_true")
    parser.add_argument("--refusal-score-type", choices=("vector_score", "fused_score", "rerank_score"), default="rerank_score")
    parser.add_argument("--refusal-score-threshold", type=float)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--index-bundle", type=Path, help="原运行对应的 AI 索引目录，仅用于人工核对记录")
    parser.add_argument("--allow-snapshot-mismatch", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = ReportConfig(
            topk=args.topk,
            default_visible=args.default_visible,
            default_stage=args.default_stage,
            page_tolerance=args.page_tolerance,
            high_match_threshold=args.high_match_threshold,
            partial_match_threshold=args.partial_match_threshold,
            count_partial_matches=args.count_partial_matches,
            refusal_score_type=args.refusal_score_type,
            refusal_score_threshold=args.refusal_score_threshold,
            allow_snapshot_mismatch=args.allow_snapshot_mismatch,
        )
        detail = generate_report(
            args.cases,
            args.results,
            args.out,
            config,
            baseline_path=args.baseline,
        )
    except SnapshotMismatchError as exc:
        print(f"索引版本校验失败：{exc}", file=sys.stderr)
        print(f"报告仍已写入：{args.out}", file=sys.stderr)
        return 4
    except (OSError, ValueError, TypeError) as exc:
        print(f"报告生成失败：{exc}", file=sys.stderr)
        return 2
    print(f"报告已生成：{Path(args.out).resolve() / 'report.html'}")
    print(f"计分题：{detail['summary']['scored_case_count']} / {detail['summary']['case_count']}")
    if detail["validation"].get("warning"):
        print(f"版本提示：{detail['validation']['warning']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
