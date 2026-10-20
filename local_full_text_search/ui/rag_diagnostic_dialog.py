"""在本机内存中展示一次 RAG 问答的完整检索过程。"""

from __future__ import annotations

import json

from PySide6.QtCore import QStandardPaths, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


def _score_summary(item: dict[str, object]) -> str:
    parts = []
    for key, label in (
        ("lexical_rank", "关键词第"),
        ("vector_rank", "向量第"),
    ):
        if item.get(key) is not None:
            parts.append(f"{label}{item[key]}")
    for key, label in (
        ("lexical_score", "BM25"),
        ("vector_score", "相似度"),
        ("fused_score", "RRF"),
        ("rerank_score", "重排"),
        ("constraint_score", "条件"),
    ):
        value = item.get(key)
        if value is not None and (key != "fused_score" or float(value) != 0):
            parts.append(f"{label} {float(value):.6f}")
    constraint = item.get("constraint_decision")
    if isinstance(constraint, dict):
        details = constraint.get("judgements")
        if isinstance(details, list):
            compact = ",".join(
                f"{value.get('kind')}:{value.get('status')}"
                for value in details
                if isinstance(value, dict)
            )
            if compact:
                parts.append(f"条件 {compact}")
    return " · ".join(parts)


class _CandidateStage(QWidget):
    def __init__(self, explanation: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._candidates: list[dict[str, object]] = []
        layout = QVBoxLayout(self)
        description = QLabel(explanation)
        description.setWordWrap(True)
        layout.addWidget(description)
        self.count_label = QLabel("等待本阶段执行")
        layout.addWidget(self.count_label)
        self.query_label = QLabel("")
        self.query_label.setWordWrap(True)
        self.query_label.setVisible(False)
        layout.addWidget(self.query_label)
        # 内容块 ID 已在详情区完整展示，不占用候选列表的横向空间。
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["本阶段排名", "排名与分数", "文件", "位置", "去留"]
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        # 排名和分数紧跟序号，窗口变窄时也能先看到最重要的诊断信息。
        # 长文件名、位置和处理原因使用固定初始宽度，避免自动扩列挤走分数。
        for column in range(4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 54)
        self.table.setColumnWidth(1, 330)
        self.table.setColumnWidth(2, 290)
        self.table.setColumnWidth(3, 140)
        layout.addWidget(self.table, 3)
        detail_label = QLabel("选中内容块的完整正文（仅在本机显示）")
        layout.addWidget(detail_label)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("点击上方候选查看原文、路径和各阶段分数。")
        layout.addWidget(self.details, 2)
        self.table.currentCellChanged.connect(self._show_selected)

    def set_candidates(self, values: list[dict[str, object]]) -> None:
        self._candidates = [dict(item) for item in values]
        self.table.setRowCount(len(self._candidates))
        self.count_label.setText(f"本阶段候选：{len(self._candidates)} 条")
        for row, item in enumerate(self._candidates):
            selected = item.get("selected")
            decision = str(item.get("reason") or "")
            if selected is False and not decision:
                decision = "未入选"
            elif selected is True and not decision:
                decision = "入选"
            if "final_rank" in item:
                rank = item.get("final_rank")
                stage_rank = str(rank) if rank is not None else "未入选"
            else:
                stage_rank = str(row + 1)
            columns = (
                stage_rank,
                _score_summary(item),
                str(item.get("filename") or ""),
                str(item.get("location") or ""),
                decision,
            )
            for column, value in enumerate(columns):
                cell = QTableWidgetItem(value)
                cell.setToolTip(value)
                self.table.setItem(row, column, cell)
        if self._candidates:
            self.table.selectRow(0)
            self._show_selected(0)
        else:
            self.details.clear()

    def reset(self) -> None:
        self._candidates = []
        self.table.setRowCount(0)
        self.count_label.setText("等待本阶段执行")
        self.query_label.clear()
        self.query_label.setVisible(False)
        self.details.clear()

    def set_query_analysis(self, terms: list[object], expression: str) -> None:
        readable = "、".join(str(item) for item in terms if str(item))
        self.query_label.setText(
            f"本次提取关键词：{readable or '无'}\nFTS 查询式：{expression or '未生成，使用短词 LIKE 回退'}"
        )
        self.query_label.setVisible(True)

    def _show_selected(self, row: int, *_unused: object) -> None:
        if not 0 <= row < len(self._candidates):
            self.details.clear()
            return
        item = self._candidates[row]
        self.details.setPlainText(
            f"文件：{item.get('filename') or ''}\n"
            f"路径：{item.get('file_path') or ''}\n"
            f"位置：{item.get('location') or ''}\n"
            f"内容块 ID：{item.get('chunk_id') or ''}\n"
            f"来源类型：{item.get('source_type') or ''}\n"
            f"排名与分数：{_score_summary(item)}\n"
            f"处理结果：{item.get('reason') or ''}\n\n"
            f"正文：\n{item.get('text') or ''}"
        )

    def export_text(self) -> str:
        """导出本阶段所有候选，而不是只导出界面当前选中的一行。"""
        sections = [self.count_label.text()]
        if self.query_label.isVisible() and self.query_label.text():
            sections.append(self.query_label.text())
        for row, item in enumerate(self._candidates, start=1):
            sections.append(
                f"#{row}\n"
                f"文件：{item.get('filename') or ''}\n"
                f"路径：{item.get('file_path') or ''}\n"
                f"位置：{item.get('location') or ''}\n"
                f"内容块 ID：{item.get('chunk_id') or ''}\n"
                f"来源类型：{item.get('source_type') or ''}\n"
                f"排名与分数：{_score_summary(item)}\n"
                f"处理结果：{item.get('reason') or ''}\n"
                f"正文：\n{item.get('text') or ''}"
            )
        return "\n\n".join(sections)


class RagDiagnosticDialog(QDialog):
    """仅界面可见；这里的正文和请求内容不会写入常规诊断日志。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("资料智能问答 · 检索诊断")
        self.resize(1180, 760)
        layout = QVBoxLayout(self)
        self.summary = QLabel("还没有提问；提问后可在此查看各阶段结果。")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.metadata = QLabel("候选数量、阈值和索引版本会在提问后显示。")
        self.metadata.setWordWrap(True)
        layout.addWidget(self.metadata)

        actions = QHBoxLayout()
        self.copy_text_button = QPushButton("复制全部诊断文本")
        self.copy_screenshot_button = QPushButton("复制全部页截图")
        self.save_screenshot_button = QPushButton("保存全部页截图")
        self.save_json_button = QPushButton("导出脱敏 JSON")
        actions.addWidget(self.copy_text_button)
        actions.addWidget(self.copy_screenshot_button)
        actions.addWidget(self.save_screenshot_button)
        actions.addWidget(self.save_json_button)
        actions.addStretch(1)
        self.export_status = QLabel(
            "截图会把所有页签的当前可见区域拼成一张长图；复制文本可获得完整候选和正文。"
        )
        self.export_status.setWordWrap(True)
        actions.addWidget(self.export_status, 1)
        layout.addLayout(actions)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.lexical = self._add_candidates(
            "关键词召回",
            "算法：QueryAnalyzer 先用固定规则去掉问句外壳、提取型号和关键词，再从 RAG 正文索引召回。"
            "较长词使用 FTS trigram，并按 BM25 排序；BM25 原始值越小越靠前。短词使用 LIKE 回退。",
        )
        self.vector = self._add_candidates(
            "向量召回",
            "算法：本地 BGE 将问题转为向量，Chroma 按余弦距离寻找语义相近的内容块。"
            "相似度显示为 1－距离；它不是答案正确率。",
        )
        self.fusion = self._add_candidates(
            "融合候选",
            "算法：用内容块 ID 合并两路命中，RRF 分数为两路 1／(k＋排名) 之和。"
            "RRF 分数用于排序，不能与向量相似度直接比较；仅前 fusion_top_k 项进入下一步。",
        )
        self.deduplicated = self._add_candidates(
            "去重结果",
            "算法：RRF 后、BGE 重排前，按内容块 ID、正文哈希和同文档三字片段相似度去除重复内容。"
            "不同文档的相同正文会保留，避免误删版本差异。",
        )
        self.rerank = self._add_candidates(
            "重排结果",
            "算法：本地 Cross-Encoder 同时阅读问题和候选正文，按相关性分数排序。"
            "此处保留完整重排候选，业务条件检查后才按最终 TopK 截断；分数不是严格概率。",
        )
        self.constraints = self._add_candidates(
            "条件检查",
            "规则：将问题中的范围、阶段、方向与正文明确表达对照，标记为 MATCH、CONFLICT 或 UNKNOWN。"
            "明确冲突的候选不能仅凭高重排分数进入最终证据。",
        )
        self.evidence = self._add_candidates(
            "证据筛选",
            "规则：必须达到重排分数阈值；另外需命中关键词，或达到向量相似度阈值。"
            "未通过的候选不会送给 LLM。",
        )
        self.context_text = self._add_text(
            "引用上下文",
            "算法：对通过筛选的正文按文本哈希去重，按长度上限截取，编号为 [S1]、[S2] 等。"
            "这里显示送入提示词的实际证据文本。",
        )
        self.request_text = self._add_text(
            "LLM 请求",
            "此处预览调用前构造的实际 SDK 请求参数。它只在本机显示，包含资料正文；"
            "不包含 API Key。若后续连接失败，预览并不代表服务商已经收到请求。",
        )
        self.query_plan_text = self._add_text(
            "问题理解",
            "这里展示确定性 QueryAnalyzer 提取出的检索词、意图和业务条件。语义检索仍使用完整问题。",
        )
        self.dedup_text = self._add_text(
            "去重明细",
            "这里显示每个被移除重复内容块对应的保留块、去重规则和相似度。",
        )
        self.draft_text = self._add_text(
            "答案初稿",
            "第一次 LLM 生成的内容只用于审查；启用二次复核时，未经复核的初稿不会直接交付。",
        )
        self.review_text = self._add_text(
            "答案复核",
            "第二次 LLM 逐条检查结论、引用和业务条件，并返回 pass、revised 或 insufficient。",
        )
        self.final_text = self._add_text(
            "最终结果",
            "这里显示本次问答最终交付给用户的答案、模型状态、警告与引用来源，"
            "可与前面的召回、重排和 LLM 请求逐级对照。",
        )
        self._request_id = ""
        self._diagnostic_payload: dict[str, object] = {"stages": {}}
        self._last_answer: object | None = None

        self.copy_text_button.clicked.connect(self.copy_all_text)
        self.copy_screenshot_button.clicked.connect(self.copy_all_tabs_screenshot)
        self.save_screenshot_button.clicked.connect(self.save_all_tabs_screenshot)
        self.save_json_button.clicked.connect(self.save_diagnostic_json)

        # 问题理解放在首位；初稿、复核和最终答案连续排列，便于顺着调用链阅读。
        for target, title in enumerate(
            ("问题理解", "答案初稿", "答案复核", "最终结果")
        ):
            current = next(
                (index for index in range(self.tabs.count()) if self.tabs.tabText(index) == title),
                -1,
            )
            if current >= 0:
                self.tabs.tabBar().moveTab(current, target)

    def _add_candidates(self, title: str, explanation: str) -> _CandidateStage:
        stage = _CandidateStage(explanation, self)
        self.tabs.addTab(stage, title)
        return stage

    def _add_text(self, title: str, explanation: str) -> QPlainTextEdit:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        label = QLabel(explanation)
        label.setWordWrap(True)
        layout.addWidget(label)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlaceholderText("尚未到达本阶段。")
        layout.addWidget(text)
        self.tabs.addTab(page, title)
        return text

    def start_query(self, request_id: str, question: str) -> None:
        self._request_id = request_id
        self._diagnostic_payload = {
            "request_id": request_id,
            "question": question,
            "stages": {},
        }
        self._last_answer = None
        self.summary.setText(f"诊断编号：{request_id}  ·  问题：{question}")
        self.metadata.setText("正在读取索引与查询参数…")
        for stage in (
            self.lexical, self.vector, self.fusion, self.deduplicated,
            self.rerank, self.constraints, self.evidence,
        ):
            stage.reset()
        self.context_text.clear()
        self.request_text.clear()
        self.query_plan_text.clear()
        self.dedup_text.clear()
        self.draft_text.clear()
        self.review_text.clear()
        self.final_text.clear()
        self.export_status.setText(
            "截图会把所有页签的当前可见区域拼成一张长图；复制文本可获得完整候选和正文。"
        )
        self.tabs.setCurrentIndex(0)

    def update_progress(self, payload: dict[str, object]) -> None:
        stages = self._diagnostic_payload.setdefault("stages", {})
        if isinstance(stages, dict):
            stage_name = str(payload.get("stage") or "unknown")
            safe_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"llm_context", "draft_answer", "reviewed_answer", "api_request", "review_request"}
            }
            for key in (
                "lexical_candidates", "vector_candidates", "fused_candidates",
                "deduplicated_candidates", "reranked_candidates",
                "constraint_candidates", "evidence_candidates",
            ):
                values = payload.get(key)
                if isinstance(values, list):
                    safe_payload[key] = [self._safe_candidate(item) for item in values]
            stages[stage_name] = safe_payload
        limits = payload.get("query_limits")
        if isinstance(limits, dict):
            self.metadata.setText(
                "查询上限：关键词 {lexical_top_k} · 向量 {vector_top_k} · "
                "融合 {fusion_top_k} · 送重排 {reranker_top_k} · 最终 {final_top_k}".format(
                    **limits
                )
            )
        if payload.get("rag_bundle_id"):
            self.metadata.setText(
                f"{self.metadata.text()}  ·  AI 索引 {payload['rag_bundle_id']}"
            )
        for key, stage in (
            ("lexical_candidates", self.lexical),
            ("vector_candidates", self.vector),
            ("fused_candidates", self.fusion),
            ("deduplicated_candidates", self.deduplicated),
            ("reranked_candidates", self.rerank),
            ("constraint_candidates", self.constraints),
            ("evidence_candidates", self.evidence),
        ):
            values = payload.get(key)
            if isinstance(values, list):
                stage.set_candidates(values)
        terms = payload.get("lexical_terms")
        if isinstance(terms, list):
            self.lexical.set_query_analysis(
                terms,
                str(payload.get("lexical_fts_expression") or ""),
            )
        query_plan = payload.get("query_plan")
        if isinstance(query_plan, dict):
            self.query_plan_text.setPlainText(
                json.dumps(query_plan, ensure_ascii=False, indent=2)
            )
        duplicate_records = payload.get("duplicate_records")
        if isinstance(duplicate_records, list):
            self.dedup_text.setPlainText(
                json.dumps(duplicate_records, ensure_ascii=False, indent=2)
            )
        draft = payload.get("draft_answer")
        if isinstance(draft, str):
            citation_check = payload.get("citation_check")
            self.draft_text.setPlainText(
                f"初稿：\n{draft}\n\n引用预检查：\n"
                f"{json.dumps(citation_check, ensure_ascii=False, indent=2)}"
            )
        review_fields = (
            "review_summary", "review_claims", "reviewed_answer", "review_verdict"
        )
        if any(key in payload for key in review_fields):
            self.review_text.setPlainText(
                "复核结论：\n"
                f"verdict：{payload.get('review_verdict') or payload.get('verdict') or ''}\n"
                f"摘要：{payload.get('review_summary') or ''}\n\n"
                "逐条判断：\n"
                f"{json.dumps(payload.get('review_claims') or [], ensure_ascii=False, indent=2)}\n\n"
                f"复核后答案：\n{payload.get('reviewed_answer') or ''}"
            )
        if payload.get("min_rerank_score") is not None:
            self.evidence.count_label.setText(
                f"{self.evidence.count_label.text()} · 重排阈值 "
                f"{float(payload['min_rerank_score']):g} · 向量阈值 "
                f"{float(payload['min_vector_score']):g}"
            )
        context = payload.get("llm_context")
        if isinstance(context, str):
            self.context_text.setPlainText(context)
        request = payload.get("api_request")
        if isinstance(request, dict):
            displayed = {
                "api_style": payload.get("api_style"),
                "provider_host": payload.get("provider_host"),
                "request": request,
            }
            self.request_text.setPlainText(json.dumps(displayed, ensure_ascii=False, indent=2))
        review_request = payload.get("review_request")
        if isinstance(review_request, dict):
            previous = self.request_text.toPlainText().strip()
            self.request_text.setPlainText(
                previous
                + "\n\n===== 第二次 LLM 审查请求（不含 API Key）=====\n"
                + json.dumps(
                    {
                        "api_style": payload.get("api_style"),
                        "provider_host": payload.get("provider_host"),
                        "request": review_request,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

    def set_final_answer(self, answer: object) -> None:
        self._last_answer = answer
        insufficient = bool(getattr(answer, "insufficient_evidence", False))
        provider = str(getattr(answer, "provider_name", "") or "")
        model = str(getattr(answer, "model_name", "") or "")
        trace_id = str(getattr(answer, "trace_id", "") or "")
        answer_text = str(getattr(answer, "answer_text", "") or "")
        warnings = tuple(getattr(answer, "warnings", ()) or ())
        sources = tuple(getattr(answer, "sources", ()) or ())
        source_ids = tuple(getattr(answer, "source_ids", ()) or ())
        status = str(getattr(getattr(answer, "status", ""), "value", getattr(answer, "status", "")))

        lines = [
            f"结果状态：{'需要补充信息' if status == 'needs_clarification' else '证据不足' if insufficient else '回答完成'}",
            f"诊断编号：{self._request_id}",
            f"调用链 Trace ID：{trace_id or '未提供'}",
            f"服务商：{provider or ('未调用 LLM' if insufficient else '未提供')}",
            f"模型：{model or ('未调用 LLM' if insufficient and not provider else '未提供')}",
        ]
        if warnings:
            lines.extend(("", "警告：", *(f"- {warning}" for warning in warnings)))
        lines.extend(("", "最终答案：", answer_text or "（空）", "", "引用来源："))
        if not sources:
            lines.append("（无）")
        for index, source in enumerate(sources, start=1):
            source_id = source_ids[index - 1] if index <= len(source_ids) else f"S{index}"
            filename = str(getattr(source, "filename", "") or "")
            location = str(getattr(source, "location_text", "") or "")
            file_path = str(getattr(source, "file_path", "") or "")
            lines.append(f"[{source_id}] {filename} · {location}\n路径：{file_path}")
        self.final_text.setPlainText("\n".join(lines))
        audit = getattr(answer, "audit", None)
        if audit is not None:
            if getattr(audit, "draft_answer", ""):
                self.draft_text.setPlainText(audit.draft_answer)
            review = getattr(audit, "review", None)
            if review is not None:
                self.review_text.setPlainText(
                    f"verdict：{review.verdict}\n摘要：{review.summary}\n\n"
                    + "\n\n".join(
                        f"结论：{claim.claim}\n引用：{', '.join(claim.citations)}\n"
                        f"状态：{claim.status}\n理由：{claim.reason}"
                        for claim in review.claims
                    )
                )

    @staticmethod
    def _safe_candidate(value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            return {}
        # 导出时不包含正文和绝对路径；用户可另用完整诊断文本主动附上正文。
        return {
            key: item
            for key, item in value.items()
            if key not in {"text", "file_path"}
        }

    def diagnostic_json(self) -> dict[str, object]:
        payload = json.loads(json.dumps(self._diagnostic_payload, ensure_ascii=False, default=str))
        answer = self._last_answer
        if answer is not None:
            sources = tuple(getattr(answer, "sources", ()) or ())
            source_ids = tuple(getattr(answer, "source_ids", ()) or ())
            payload["result"] = {
                "status": str(
                    getattr(
                        getattr(answer, "status", ""),
                        "value",
                        getattr(answer, "status", ""),
                    )
                ),
                "answer": str(getattr(answer, "answer_text", "") or ""),
                "trace_id": str(getattr(answer, "trace_id", "") or ""),
                "provider": str(getattr(answer, "provider_name", "") or ""),
                "model": str(getattr(answer, "model_name", "") or ""),
                "source_ids": list(source_ids),
                "sources": [
                    {
                        "source_id": source_ids[index] if index < len(source_ids) else f"S{index + 1}",
                        "filename": str(getattr(source, "filename", "") or ""),
                        "location": str(getattr(source, "location_text", "") or ""),
                    }
                    for index, source in enumerate(sources)
                ],
                "warnings": list(getattr(answer, "warnings", ()) or ()),
            }
            audit = getattr(answer, "audit", None)
            if audit is not None:
                review = getattr(audit, "review", None)
                payload["result"]["audit"] = {
                    "draft_answer": str(getattr(audit, "draft_answer", "") or ""),
                    "review_verdict": str(getattr(review, "verdict", "") or ""),
                    "review_summary": str(getattr(review, "summary", "") or ""),
                    "claims": [
                        {
                            "claim": item.claim,
                            "citations": list(item.citations),
                            "status": item.status,
                            "reason": item.reason,
                        }
                        for item in (getattr(review, "claims", ()) or ())
                    ],
                }
        payload["export_note"] = "已移除候选正文、绝对文件路径、API Key 和完整 API 请求体。"
        return payload

    def save_diagnostic_json(self) -> None:
        pictures = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.PicturesLocation
        )
        safe_request_id = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in (self._request_id or "rag-diagnostic")
        )
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出脱敏检索诊断",
            f"{pictures}/{safe_request_id}-检索诊断.json",
            "JSON 文件 (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(self.diagnostic_json(), stream, ensure_ascii=False, indent=2)
            self.export_status.setText(f"脱敏诊断已保存：{path}")
        except OSError as exc:
            self.export_status.setText(f"诊断导出失败：{exc}")

    def diagnostic_text(self) -> str:
        """生成适合粘贴到缺陷单或聊天中的完整诊断文本。"""
        sections = [
            "资料智能问答 · 检索诊断",
            self.summary.text(),
            self.metadata.text(),
        ]
        candidate_pages = {
            self.lexical: self.lexical.export_text,
            self.vector: self.vector.export_text,
            self.fusion: self.fusion.export_text,
            self.deduplicated: self.deduplicated.export_text,
            self.rerank: self.rerank.export_text,
            self.constraints: self.constraints.export_text,
            self.evidence: self.evidence.export_text,
        }
        text_pages = {
            self.context_text: self.context_text.toPlainText,
            self.request_text: self.request_text.toPlainText,
            self.query_plan_text: self.query_plan_text.toPlainText,
            self.dedup_text: self.dedup_text.toPlainText,
            self.draft_text: self.draft_text.toPlainText,
            self.review_text: self.review_text.toPlainText,
            self.final_text: self.final_text.toPlainText,
        }
        for index in range(self.tabs.count()):
            page = self.tabs.widget(index)
            title = self.tabs.tabText(index)
            getter = candidate_pages.get(page) or text_pages.get(
                next(
                    (
                        text_widget
                        for text_widget in text_pages
                        if text_widget.parentWidget() is page
                    ),
                    None,
                )
            )
            body = getter() if getter is not None else ""
            sections.append(f"\n===== {title} =====\n{body or '（无数据）'}")
        return "\n".join(sections)

    def copy_all_text(self) -> None:
        QApplication.clipboard().setText(self.diagnostic_text())
        self.export_status.setText("已复制完整诊断文本，可直接粘贴发送。")

    def capture_all_tabs(self) -> QPixmap:
        """把每个诊断页签当前可见区域纵向拼接成一张长图。"""
        current_index = self.tabs.currentIndex()
        captured: list[tuple[str, QPixmap]] = []
        try:
            for index in range(self.tabs.count()):
                self.tabs.setCurrentIndex(index)
                QApplication.processEvents()
                page = self.tabs.widget(index)
                captured.append((self.tabs.tabText(index), page.grab()))
        finally:
            self.tabs.setCurrentIndex(current_index)
            QApplication.processEvents()

        margin = 20
        title_height = 38
        header_height = 112
        content_width = max((pixmap.width() for _, pixmap in captured), default=900)
        width = content_width + margin * 2
        height = header_height + margin + sum(
            title_height + pixmap.height() + margin for _, pixmap in captured
        )
        result = QPixmap(width, max(height, 1))
        result.fill(QColor("white"))
        painter = QPainter(result)
        painter.setPen(QColor("#111827"))
        title_font = QFont(painter.font())
        title_font.setPointSize(14)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(margin, 30, "资料智能问答 · 完整检索诊断")
        body_font = QFont(painter.font())
        body_font.setPointSize(9)
        body_font.setBold(False)
        painter.setFont(body_font)
        painter.drawText(
            margin,
            42,
            width - margin * 2,
            header_height - 38,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
            f"{self.summary.text()}\n{self.metadata.text()}",
        )

        y = header_height
        tab_font = QFont(body_font)
        tab_font.setPointSize(11)
        tab_font.setBold(True)
        for title, pixmap in captured:
            painter.fillRect(margin, y, content_width, title_height, QColor("#e8f0fa"))
            painter.setFont(tab_font)
            painter.drawText(
                margin + 10,
                y,
                content_width - 20,
                title_height,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                title,
            )
            y += title_height
            painter.drawPixmap(margin, y, pixmap)
            y += pixmap.height() + margin
        painter.end()
        return result

    def copy_all_tabs_screenshot(self) -> None:
        pixmap = self.capture_all_tabs()
        QApplication.clipboard().setPixmap(pixmap)
        self.export_status.setText(
            f"已复制全部 {self.tabs.count()} 个页签的长截图（{pixmap.width()}×{pixmap.height()}），可直接粘贴发送。"
        )

    def save_all_tabs_screenshot(self) -> None:
        pictures = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.PicturesLocation
        )
        safe_request_id = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in (self._request_id or "rag-diagnostic")
        )
        default_path = f"{pictures}/{safe_request_id}-完整检索诊断.png"
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存完整检索诊断截图",
            default_path,
            "PNG 图片 (*.png)",
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        pixmap = self.capture_all_tabs()
        if pixmap.save(path, "PNG"):
            self.export_status.setText(f"完整诊断截图已保存：{path}")
        else:
            self.export_status.setText(f"截图保存失败，请检查目录权限：{path}")

    def mark_finished(self, status: str) -> None:
        self.summary.setText(f"{self.summary.text()}  ·  {status}")
        if not self.final_text.toPlainText().strip():
            self.final_text.setPlainText(
                f"结果状态：{status}\n"
                f"诊断编号：{self._request_id}\n\n"
                "本次任务没有生成最终答案；请结合前面最后一个有数据的阶段定位问题。"
            )
