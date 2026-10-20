"""RAG 评测诊断报告。

报告层只读取已经保存的评测结果，不启动模型、不发起网络请求，也不改变检索算法。
"""

from .matching import EvidenceMatch, MatchConfig, match_quote, normalize_text
from .service import ReportConfig, build_report_detail, generate_report

__all__ = [
    "EvidenceMatch",
    "MatchConfig",
    "ReportConfig",
    "build_report_detail",
    "generate_report",
    "match_quote",
    "normalize_text",
]
