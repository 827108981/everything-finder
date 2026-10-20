from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EXPECTED_BEHAVIORS = {"answer", "partial_answer", "clarify", "refuse"}


class GoldDatasetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GoldCase:
    case_id: str
    question: str
    expected_behavior: str
    required_facts: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    constraints: dict[str, Any]
    missing_fields: tuple[str, ...]
    gold_evidence: tuple[dict[str, Any], ...]
    hard_negatives: tuple[dict[str, Any], ...]
    group_id: str
    corpus_snapshot: str
    review_status: str
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GoldDataset:
    path: Path
    sha256: str
    cases: tuple[GoldCase, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """读取 JSONL 或审核人员使用的 JSON 容器。

    评测报告需要保留 ``_阅读说明`` 等顶层说明，不能把 JSON 文件粗暴地
    转成 JSONL 后丢失索引目录和审核约束。
    """

    text = path.read_text(encoding="utf-8-sig")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GoldDatasetError(f"第 {line_number} 行不是有效 JSON：{exc.msg}") from exc
            if not isinstance(row, dict):
                raise GoldDatasetError(f"第 {line_number} 行必须是 JSON 对象")
            rows.append(row)
        return rows, {}
    if isinstance(document, dict) and isinstance(document.get("cases"), list):
        metadata = {key: value for key, value in document.items() if key != "cases"}
        return [row for row in document["cases"] if isinstance(row, dict)], metadata
    if isinstance(document, list):
        return [row for row in document if isinstance(row, dict)], {}
    # 单行 JSONL 恰好也是一个 JSON 对象；它仍然代表一条用例。
    if isinstance(document, dict):
        return [document], {}
    raise GoldDatasetError("测试集必须是 JSONL，或包含 cases 数组的 JSON 文件")


def load_gold_dataset(path: Path, *, reviewed_only: bool = True) -> GoldDataset:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise GoldDatasetError(f"找不到测试集文件：{path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cases: list[GoldCase] = []
    seen: set[str] = set()
    errors: list[str] = []
    try:
        rows, metadata = _load_rows(path)
    except GoldDatasetError as exc:
        raise
    for line_number, row in enumerate(rows, 1):
        case_id = str(row.get("id") or "").strip()
        if not case_id:
            errors.append(f"第 {line_number} 行缺少 id")
            continue
        if case_id in seen:
            errors.append(f"题号重复：{case_id}")
            continue
        seen.add(case_id)
        error_count_before = len(errors)
        expected = str(row.get("expected_behavior") or "").strip()
        question = str(row.get("question") or "").strip()
        if not question:
            errors.append(f"{case_id}：question 不能为空")
        if expected not in EXPECTED_BEHAVIORS:
            errors.append(f"{case_id}：不支持的 expected_behavior：{expected}")
        status = str(row.get("review_status") or "").strip()
        if reviewed_only and status != "reviewed":
            errors.append(f"{case_id}：尚未标记为 reviewed")
        if not isinstance(row.get("constraints", {}), dict):
            errors.append(f"{case_id}：constraints 必须是对象")
        list_fields = ("required_facts", "forbidden_claims", "missing_fields", "gold_evidence", "hard_negatives")
        for field in list_fields:
            if not isinstance(row.get(field, []), list):
                errors.append(f"{case_id}：{field} 必须是数组")
        evidence = row.get("gold_evidence", [])
        for index, item in enumerate(evidence if isinstance(evidence, list) else [], 1):
            if not isinstance(item, dict) or not item.get("file") or not item.get("quote"):
                errors.append(f"{case_id}：gold_evidence 第 {index} 条缺少 file 或 quote")
        negatives = row.get("hard_negatives", [])
        for index, item in enumerate(negatives if isinstance(negatives, list) else [], 1):
            if not isinstance(item, dict) or not item.get("file") or not item.get("quote"):
                errors.append(f"{case_id}：hard_negatives 第 {index} 条缺少 file 或 quote")
        if expected != "refuse" and not evidence:
            errors.append(f"{case_id}：非拒答题至少需要一条 gold_evidence")
        if expected == "refuse" and evidence:
            errors.append(f"{case_id}：拒答题不应把证据列为正向 Gold")
        if len(errors) > error_count_before:
            continue
        cases.append(
            GoldCase(
                case_id=case_id,
                question=question,
                expected_behavior=expected,
                required_facts=tuple(str(value) for value in row.get("required_facts", [])),
                forbidden_claims=tuple(str(value) for value in row.get("forbidden_claims", [])),
                constraints=dict(row.get("constraints") or {}),
                missing_fields=tuple(str(value) for value in row.get("missing_fields", [])),
                gold_evidence=tuple(dict(value) for value in row.get("gold_evidence", [])),
                hard_negatives=tuple(dict(value) for value in row.get("hard_negatives", [])),
                group_id=str(row.get("group_id") or case_id),
                corpus_snapshot=str(row.get("corpus_snapshot") or ""),
                review_status=status,
                raw=row,
            )
        )
    if errors:
        raise GoldDatasetError("测试集校验失败：\n" + "\n".join(errors))
    if not cases:
        raise GoldDatasetError("测试集为空")
    return GoldDataset(path=path, sha256=digest, cases=tuple(cases), metadata=metadata)
