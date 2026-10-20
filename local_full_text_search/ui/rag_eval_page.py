from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from local_full_text_search.config.constants import APP_DATA_DIR
from local_full_text_search.core.open_location import open_file
from local_full_text_search.rag.evaluation.gold_loader import GoldDatasetError, load_gold_dataset
from local_full_text_search.rag.evaluation.metrics import (
    METRIC_DEFINITIONS,
    STAGE_ORDER,
    STAGE_LABELS,
    build_rank_history,
)
from local_full_text_search.rag.evaluation.runner import RagEvalConfig


class RagEvalPage(QWidget):
    start_requested = Signal(object)
    preflight_requested = Signal(object)
    stop_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Page")
        self._case_results: dict[str, dict[str, Any]] = {}
        self._run_dir = ""
        self._busy = False
        self._build_ui()
        self._load_default_dataset()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(26, 22, 26, 20)
        root.setSpacing(12)
        title = QLabel("RAG 批量评测")
        title.setObjectName("PageTitle")
        intro = QLabel("导入已审核的 Gold 测试集，逐题检查正确证据在各检索阶段的出现情况，并对照最终问答结果。")
        intro.setWordWrap(True)
        intro.setObjectName("MutedText")
        root.addWidget(title)
        root.addWidget(intro)

        config_card = QFrame()
        config_card.setObjectName("Card")
        config_layout = QVBoxLayout(config_card)
        config_layout.setContentsMargins(14, 12, 14, 12)
        config_layout.setSpacing(8)
        dataset_row = QHBoxLayout()
        dataset_row.addWidget(QLabel("测试集"))
        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText("选择 JSONL Gold 测试集")
        dataset_row.addWidget(self.dataset_path, 1)
        self.browse_dataset_button = QPushButton("选择文件…")
        self.browse_dataset_button.clicked.connect(self._choose_dataset)
        dataset_row.addWidget(self.browse_dataset_button)
        config_layout.addLayout(dataset_row)

        options = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItem("只测检索（不调用 LLM）", "retrieval-only")
        self.mode.addItem("完整问答（会调用已配置的 LLM）", "full")
        self.mode.currentIndexChanged.connect(self._update_mode_note)
        options.addWidget(QLabel("运行模式"))
        options.addWidget(self.mode)
        options.addSpacing(12)
        options.addWidget(QLabel("题号筛选"))
        self.case_filter = QLineEdit()
        self.case_filter.setPlaceholderText("留空跑全部；多个题号用逗号分隔")
        options.addWidget(self.case_filter, 1)
        config_layout.addLayout(options)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("报告目录"))
        self.output_path = QLineEdit(str(APP_DATA_DIR / "rag_eval_runs"))
        output_row.addWidget(self.output_path, 1)
        self.browse_output_button = QPushButton("选择目录…")
        self.browse_output_button.clicked.connect(self._choose_output)
        output_row.addWidget(self.browse_output_button)
        config_layout.addLayout(output_row)
        self.dataset_status = QLabel("请选择测试集。")
        self.dataset_status.setWordWrap(True)
        config_layout.addWidget(self.dataset_status)
        self.mode_note = QLabel()
        self.mode_note.setWordWrap(True)
        config_layout.addWidget(self.mode_note)
        root.addWidget(config_card)

        action_row = QHBoxLayout()
        self.validate_button = QPushButton("检查测试集与索引")
        self.validate_button.clicked.connect(self._request_preflight)
        self.start_button = QPushButton("开始评测")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.clicked.connect(self._request_start)
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.open_report_button = QPushButton("打开报告目录")
        self.open_report_button.setEnabled(False)
        self.open_report_button.clicked.connect(self._open_report_dir)
        self.status_label = QLabel("等待运行")
        self.status_label.setObjectName("MutedText")
        action_row.addWidget(self.validate_button)
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.stop_button)
        action_row.addWidget(self.open_report_button)
        action_row.addWidget(self.status_label, 1)
        root.addLayout(action_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self.overview = QPlainTextEdit()
        self.overview.setReadOnly(True)
        self.overview.setPlaceholderText("索引预检和评测汇总会显示在这里。")
        self.tabs.addTab(self.overview, "概览")

        cases_panel = QWidget()
        cases_layout = QVBoxLayout(cases_panel)
        cases_layout.setContentsMargins(6, 8, 6, 6)
        self.case_table = self._table(("题号", "预期行为", "实际行为", "行为匹配", "Gold 首次出现", "首位排名", "耗时"))
        self.case_table.itemSelectionChanged.connect(self._show_selected_case)
        cases_layout.addWidget(self.case_table)
        self.tabs.addTab(cases_panel, "逐题结果")

        self.stage_tabs = QTabWidget()
        for stage in STAGE_ORDER:
            stage_panel = QWidget()
            stage_layout = QVBoxLayout(stage_panel)
            stage_layout.setContentsMargins(6, 8, 6, 6)
            explanation = QLabel(self._stage_explanation(stage))
            explanation.setWordWrap(True)
            explanation.setObjectName("MutedText")
            stage_layout.addWidget(explanation)
            splitter = QSplitter(Qt.Orientation.Vertical)
            table = self._table(("阶段名次", "Gold/反例", "文件", "位置", "关键词名次", "向量名次", "向量分数", "RRF分数", "重排分数", "处理结果"))
            header = table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            header.setStretchLastSection(False)
            for column, width in enumerate((76, 108, 210, 150, 96, 96, 108, 108, 108, 260)):
                table.setColumnWidth(column, width)
            table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            detail = QPlainTextEdit()
            detail.setReadOnly(True)
            detail.setPlaceholderText("选择候选后显示正文和淘汰原因。")
            table.itemSelectionChanged.connect(lambda s=stage, t=table, d=detail: self._show_candidate_detail(s, t, d))
            splitter.addWidget(table)
            splitter.addWidget(detail)
            splitter.setSizes([350, 180])
            stage_layout.addWidget(splitter, 1)
            self.stage_tabs.addTab(stage_panel, STAGE_LABELS[stage])
            setattr(self, f"{stage}_table", table)
        stages_panel = QWidget()
        stages_layout = QVBoxLayout(stages_panel)
        stages_layout.setContentsMargins(4, 4, 4, 4)
        stages_layout.addWidget(self.stage_tabs)
        self.tabs.addTab(stages_panel, "检索链路")

        answer_panel = QWidget()
        answer_layout = QVBoxLayout(answer_panel)
        answer_layout.setContentsMargins(8, 8, 8, 8)
        self.answer_detail = QPlainTextEdit()
        self.answer_detail.setReadOnly(True)
        self.answer_detail.setPlaceholderText("选择题目后显示最终答案、引用和待人工核对项。")
        answer_layout.addWidget(self.answer_detail, 1)
        behavior_row = QHBoxLayout()
        behavior_row.addWidget(QLabel("人工行为判定"))
        self.behavior_review = QComboBox()
        self.behavior_review.addItems(["待审核", "符合预期", "不符合预期", "应部分回答", "应澄清", "应拒答"])
        behavior_row.addWidget(self.behavior_review)
        behavior_row.addStretch(1)
        answer_layout.addLayout(behavior_row)
        self.fact_table = self._table(("必需事实", "字面命中提示", "人工结论"))
        self.fact_table.setMaximumHeight(190)
        answer_layout.addWidget(self.fact_table)
        self.save_review_button = QPushButton("保存人工核对")
        self.save_review_button.setEnabled(False)
        self.save_review_button.clicked.connect(self._save_manual_review)
        answer_layout.addWidget(self.save_review_button, alignment=Qt.AlignmentFlag.AlignRight)
        self.tabs.addTab(answer_panel, "答案核对")

        history_panel = QWidget()
        history_layout = QVBoxLayout(history_panel)
        history_layout.setContentsMargins(8, 8, 8, 8)
        self.history_old_path = QLineEdit()
        self.history_new_path = QLineEdit()
        old_row = QHBoxLayout()
        old_row.addWidget(QLabel("基线运行"))
        old_row.addWidget(self.history_old_path, 1)
        old_browse = QPushButton("选择…")
        old_browse.clicked.connect(lambda: self._choose_history_dir(self.history_old_path))
        old_row.addWidget(old_browse)
        history_layout.addLayout(old_row)
        new_row = QHBoxLayout()
        new_row.addWidget(QLabel("对比运行"))
        new_row.addWidget(self.history_new_path, 1)
        new_browse = QPushButton("选择…")
        new_browse.clicked.connect(lambda: self._choose_history_dir(self.history_new_path))
        new_row.addWidget(new_browse)
        self.compare_button = QPushButton("比较两次运行")
        self.compare_button.clicked.connect(self._compare_runs)
        new_row.addWidget(self.compare_button)
        history_layout.addLayout(new_row)
        self.history_result = QPlainTextEdit()
        self.history_result.setReadOnly(True)
        self.history_result.setPlaceholderText("选择两次运行的报告目录后，比较相同 Gold 集、索引和配置下的阶段指标变化。")
        history_layout.addWidget(self.history_result, 1)
        self.tabs.addTab(history_panel, "历史对比")

        self._update_mode_note()

    @staticmethod
    def _table(headers: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(list(headers))
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setMinimumSectionSize(80)
        table.verticalHeader().setVisible(False)
        return table

    def _load_default_dataset(self) -> None:
        fixture = Path(__file__).resolve().parents[2] / "tests" / "rag" / "fixtures" / "rag_gold_v1.jsonl"
        if fixture.is_file():
            self.dataset_path.setText(str(fixture))
            self._refresh_dataset_preview()
        else:
            self._refresh_dataset_preview()

    def _choose_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 Gold 测试集", self.dataset_path.text(), "JSONL 文件 (*.jsonl);;所有文件 (*.*)")
        if path:
            self.dataset_path.setText(path)
            self._refresh_dataset_preview()

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择评测报告保存目录", self.output_path.text())
        if path:
            self.output_path.setText(path)

    def _refresh_dataset_preview(self) -> bool:
        try:
            dataset = load_gold_dataset(Path(self.dataset_path.text().strip()))
        except (GoldDatasetError, OSError) as exc:
            self.dataset_status.setText(f"测试集不可用：{exc}")
            return False
        counts: dict[str, int] = {}
        for case in dataset.cases:
            counts[case.expected_behavior] = counts.get(case.expected_behavior, 0) + 1
        count_text = "、".join(f"{key} {value} 题" for key, value in counts.items())
        self.dataset_status.setText(
            f"已加载 {len(dataset.cases)} 题，审核状态全部有效；{count_text}。文件 SHA-256：{dataset.sha256[:12]}…"
        )
        return True

    def _update_mode_note(self) -> None:
        if self.mode.currentData() == "full":
            self.mode_note.setText("完整问答会逐题调用设置中已保存的 LLM 配置；如果启用了答案二次复核，每题还会产生复核请求。不会在界面显示或写出 API Key。")
        else:
            self.mode_note.setText("只测本地分析、关键词/向量召回、融合、重排与证据筛选，不读取 API Key，也不调用 LLM。")

    def _config(self) -> RagEvalConfig:
        ids = tuple(value.strip() for value in self.case_filter.text().replace("\n", ",").split(",") if value.strip())
        output = self.output_path.text().strip()
        return RagEvalConfig(
            dataset_path=Path(self.dataset_path.text().strip()),
            mode=str(self.mode.currentData()),
            case_ids=ids,
            output_root=Path(output) if output else None,
        )

    def _request_preflight(self) -> None:
        if not self._refresh_dataset_preview():
            return
        self.preflight_requested.emit(self._config())

    def _request_start(self) -> None:
        if not self._refresh_dataset_preview():
            return
        config = self._config()
        if config.mode == "full":
            answer = QMessageBox.question(
                self,
                "确认运行完整问答评测",
                "将使用已保存的 LLM 配置逐题生成答案。启用了二次复核时，每题还会追加一次 API 请求。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._case_results.clear()
        self._run_dir = ""
        self.case_table.setRowCount(0)
        self.progress.setValue(0)
        self.start_requested.emit(config)

    def set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = bool(busy)
        self.start_button.setEnabled(not busy)
        self.validate_button.setEnabled(not busy)
        self.browse_dataset_button.setEnabled(not busy)
        self.browse_output_button.setEnabled(not busy)
        self.mode.setEnabled(not busy)
        self.case_filter.setEnabled(not busy)
        self.output_path.setEnabled(not busy)
        self.dataset_path.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        if message:
            self.status_label.setText(message)

    def show_progress(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        event_type = str(payload.get("type") or "")
        if event_type == "preflight_complete":
            coverage = payload.get("coverage") or {}
            self.overview.setPlainText(
                "測試集與 AI 索引檢查通過\n"
                f"AI 索引：{payload.get('bundle_id', '')}\n"
                f"題數：{coverage.get('case_count', 0)}（已審核 {coverage.get('reviewed_case_count', 0)}）\n"
                f"Gold 正證據定位：{coverage.get('positive_evidence_located', 0)}/{coverage.get('positive_evidence_total', 0)}\n"
                f"需要人工確認的證據映射：{coverage.get('evidence_needing_review', 0)}\n"
                f"拒答題：{coverage.get('refusal_case_count', 0)}"
            )
            self.status_label.setText("测试集与索引检查完成")
            return
        if event_type in {"run_started", "run_finished", "run_partial"}:
            if event_type == "run_started":
                self._run_dir = str(payload.get("run_dir") or "")
                self.overview.setPlainText(
                    f"评测运行中\n运行编号：{payload.get('run_id', '')}\n报告目录：{self._run_dir}\n总题数：{payload.get('total', 0)}"
                )
            elif event_type in {"run_finished", "run_partial"}:
                self.show_run_result(payload)
            return
        if event_type == "case_finished":
            result = payload.get("result")
            if isinstance(result, dict):
                self._case_results[str(result.get("id") or "")] = result
                self._append_case_row(result)
            total = max(1, int(payload.get("total") or 1))
            self.progress.setValue(int(int(payload.get("index") or 0) * 100 / total))
            self.status_label.setText(f"已完成 {payload.get('index', 0)}/{total}：{payload.get('result', {}).get('id', '')}")
            return
        if event_type == "stage":
            stage = str(payload.get("stage") or payload.get("message") or "处理中")
            self.status_label.setText(f"{payload.get('case_id', '')} · {stage}")
            return
        if event_type == "cancelled":
            self.status_label.setText("正在收尾并保存已完成题目…")
            if isinstance(payload.get("partial_result"), dict):
                self.show_run_result(payload["partial_result"], cancelled=True)

    def show_run_result(self, report: dict[str, Any], *, cancelled: bool = False) -> None:
        self._run_dir = str(report.get("run_dir") or self._run_dir)
        summary = report.get("summary") or {}
        if cancelled or report.get("state") == "cancelled":
            state = "已停止（已完成题目与部分报告已保存）"
        elif report.get("state") == "partial":
            state = "部分完成（存在失败题目，详见逐题结果）"
        else:
            state = "评测完成"
        lines = [
            state,
            f"运行编号：{report.get('run_id', '')}",
            f"报告目录：{self._run_dir}",
            f"完成题数：{summary.get('completed_case_count', 0)}/{summary.get('case_count', 0)}",
        ]
        if summary.get("behavior_accuracy") is not None:
            lines.append(f"完整问答行为匹配率：{summary['behavior_accuracy'] * 100:.1f}%")
        lines.append("\n阶段指标（Gold 命中率 / 平均证据召回率 / MRR / 反例候选数）：")
        for stage in STAGE_ORDER:
            values = summary.get("stages", {}).get(stage, {})
            hit = _percent(values.get("gold_hit_rate"))
            recall = _percent(values.get("mean_evidence_recall"))
            mrr = _percent(values.get("mean_reciprocal_rank"))
            lines.append(f"{STAGE_LABELS[stage]}：{hit} / {recall} / {mrr} / {values.get('hard_negative_candidate_count', 0)}")
        lines.append("\n具体候选、正文和每题失败阶段请查看“逐题结果”和“检索链路”。")
        lines.extend(["", "指标解释："])
        lines.extend(f"- {text}" for text in METRIC_DEFINITIONS.values())
        lines.append(f"本次 API 请求数：{summary.get('api_call_count', 0)}")
        self.overview.setPlainText("\n".join(lines))
        self.open_report_button.setEnabled(bool(self._run_dir and Path(self._run_dir).is_dir()))
        self._refresh_history_defaults()
        for result in report.get("results", []):
            if isinstance(result, dict) and str(result.get("id") or "") not in self._case_results:
                self._case_results[str(result.get("id") or "")] = result
                self._append_case_row(result)
        self.status_label.setText(state)
        self.progress.setValue(100 if not cancelled else self.progress.value())

    def show_failure(self, message: str) -> None:
        self.status_label.setText("评测失败")
        self.overview.setPlainText(f"评测失败\n\n{message}")

    def show_cancelled(self) -> None:
        self.status_label.setText("评测已停止；已完成题目及报告已保留")

    def show_finished(self) -> None:
        self.set_busy(False)

    def _append_case_row(self, result: dict[str, Any]) -> None:
        row = self.case_table.rowCount()
        self.case_table.insertRow(row)
        first = sorted(
            set(result.get("first_gold_hit", {}).values()),
            key=lambda item: STAGE_ORDER.index(item) if item in STAGE_ORDER else len(STAGE_ORDER),
        )
        match = result.get("behavior_match")
        stage_values = result.get("stages", {})
        first_rank = None
        for stage in STAGE_ORDER:
            first_rank = stage_values.get(stage, {}).get("first_gold_rank")
            if first_rank is not None:
                break
        values = (
            str(result.get("id") or ""),
            str(result.get("expected_behavior") or ""),
            str(result.get("actual_behavior") or ""),
            "待检索模式" if match is None else ("符合" if match else "不符合"),
            first[0] if first else ("无正证据" if result.get("expected_behavior") == "refuse" else "未命中"),
            str(first_rank if first_rank is not None else "—"),
            f"{int(result.get('duration_ms') or 0)} ms",
        )
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column == 0:
                item.setData(Qt.ItemDataRole.UserRole, str(result.get("id") or ""))
            self.case_table.setItem(row, column, item)
        self.case_table.resizeColumnsToContents()

    def _show_selected_case(self) -> None:
        selected = self.case_table.selectedItems()
        if not selected:
            return
        case_id = str(self.case_table.item(selected[0].row(), 0).text())
        result = self._case_results.get(case_id)
        if not result:
            return
        for stage in STAGE_ORDER:
            self._fill_stage_table(stage, result)
        answer = result.get("answer") or {}
        answer_lines = [
            f"题号：{case_id}",
            f"预期行为：{result.get('expected_behavior', '')}；实际行为：{result.get('actual_behavior', '')}",
            f"状态：{answer.get('status', '未运行')}",
            f"模型：{answer.get('model', '未调用')}",
            "",
            "答案：",
            str(answer.get("text") or "（只测检索模式未生成答案）"),
            "",
            "引用来源：",
        ]
        for source in answer.get("sources", []):
            if isinstance(source, dict):
                answer_lines.append(f"[{source.get('source_id', '')}] {source.get('filename', '')} · {source.get('location', '')}")
        answer_lines.extend(["", "必需事实字面命中提示（仅供定位，不代表语义判分）："])
        hints = result.get("required_fact_text_hints") or {}
        for fact in hints.get("matched", []):
            answer_lines.append(f"✓ {fact}")
        for fact in hints.get("unmatched", []):
            answer_lines.append(f"○ {fact}")
        answer_lines.extend(["", "禁说结论命中："])
        answer_lines.extend(str(item) for item in result.get("forbidden_claims_found", []) or ["无字面命中"])
        self.answer_detail.setPlainText("\n".join(answer_lines))
        review_data = self._load_case_review(case_id)
        behavior_verdict = str(review_data.get("behavior_verdict") or "待审核")
        behavior_index = self.behavior_review.findText(behavior_verdict)
        self.behavior_review.setCurrentIndex(max(0, behavior_index))
        self._fill_fact_table(result, review_data)
        self.save_review_button.setEnabled(bool(self._run_dir and result.get("required_facts")))
        self.tabs.setCurrentIndex(2)

    def _fill_stage_table(self, stage: str, result: dict[str, Any]) -> None:
        table: QTableWidget = getattr(self, f"{stage}_table")
        table.setRowCount(0)
        candidates = result.get("candidate_stages", {}).get(stage, [])
        positive_ids = {
            chunk_id for item in result.get("gold_mappings", []) if item.get("kind") == "positive"
            for chunk_id in item.get("chunk_ids", [])
        }
        negative_ids = {
            chunk_id for item in result.get("gold_mappings", []) if item.get("kind") == "hard_negative"
            for chunk_id in item.get("chunk_ids", [])
        }
        review_ids = {
            chunk_id for item in result.get("gold_mappings", []) if item.get("confidence") == "review"
            for chunk_id in item.get("chunk_ids", [])
        }
        rank_history = build_rank_history(result.get("candidate_stages", {}))
        for index, candidate in enumerate(candidates, 1):
            chunk_id = str(candidate.get("chunk_id") or "")
            labels = []
            if chunk_id in review_ids:
                labels.append("待确认")
            if chunk_id in positive_ids:
                labels.append("正确证据")
            if chunk_id in negative_ids:
                labels.append("指定反例")
            label = " + ".join(labels) if labels else "未判定"
            if stage == "evidence":
                rank = candidate.get("final_rank")
                rank_text = str(rank) if rank is not None else "未入选"
            else:
                rank_text = str(index)
            values = (
                rank_text, label, str(candidate.get("filename") or ""), str(candidate.get("location") or ""),
                _value(candidate.get("lexical_rank")), _value(candidate.get("vector_rank")),
                _value(candidate.get("vector_score")),
                _value(candidate.get("fused_score")), _value(candidate.get("rerank_score")),
                str(candidate.get("reason") or candidate.get("constraint_decision", {}).get("reason") or ""),
            )
            row = table.rowCount()
            table.insertRow(row)
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if column == 0:
                    details_candidate = dict(candidate)
                    details_candidate["_stage_rank"] = rank_text
                    details_candidate["_rank_history"] = rank_history.get(chunk_id, [])
                    cell.setData(Qt.ItemDataRole.UserRole, details_candidate)
                if column == 1:
                    cell.setToolTip(chunk_id)
                table.setItem(row, column, cell)

    @staticmethod
    def _show_candidate_detail(stage: str, table: QTableWidget, detail: QPlainTextEdit) -> None:
        selected = table.selectedItems()
        if not selected:
            return
        candidate = table.item(selected[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        if isinstance(candidate, dict):
            detail.setPlainText(
                f"阶段：{STAGE_LABELS.get(stage, stage)}\n"
                f"本阶段名次：{candidate.get('_stage_rank', '')}\n"
                f"文件：{candidate.get('filename', '')}\n位置：{candidate.get('location', '')}\n"
                f"跨阶段排名轨迹：{_format_rank_history(candidate.get('_rank_history', []))}\n"
                f"关键词名次：{_value(candidate.get('lexical_rank'))}；"
                f"向量名次：{_value(candidate.get('vector_rank'))}；"
                f"向量相似度：{_value(candidate.get('vector_score'))}；"
                f"RRF：{_value(candidate.get('fused_score'))}；"
                f"重排分数：{_value(candidate.get('rerank_score'))}\n"
                f"内容块 ID：{candidate.get('chunk_id', '')}\n"
                f"处理原因：{candidate.get('reason') or candidate.get('constraint_decision', {}).get('reason', '')}\n\n"
                f"正文：\n{candidate.get('text', '')}"
            )

    def _fill_fact_table(self, result: dict[str, Any], review_data: dict[str, Any] | None = None) -> None:
        facts = list(result.get("required_facts") or [])
        hints = result.get("required_fact_text_hints") or {}
        matched = set(hints.get("matched", []))
        saved_facts = {
            str(item.get("fact") or ""): str(item.get("verdict") or "待审核")
            for item in (review_data or {}).get("required_facts", [])
            if isinstance(item, dict)
        }
        self.fact_table.setRowCount(len(facts))
        for row, fact in enumerate(facts):
            self.fact_table.setItem(row, 0, QTableWidgetItem(str(fact)))
            self.fact_table.setItem(row, 1, QTableWidgetItem("字面命中" if fact in matched else "未发现字面命中"))
            verdict = QComboBox()
            verdict.addItems(["待审核", "支持", "部分支持", "未覆盖"])
            verdict.setProperty("fact", str(fact))
            saved_index = verdict.findText(saved_facts.get(str(fact), "待审核"))
            verdict.setCurrentIndex(max(0, saved_index))
            self.fact_table.setCellWidget(row, 2, verdict)
        self.fact_table.resizeColumnsToContents()

    def _save_manual_review(self) -> None:
        selected = self.case_table.selectedItems()
        if not selected or not self._run_dir:
            return
        case_id = str(self.case_table.item(selected[0].row(), 0).text())
        verdicts = []
        for row in range(self.fact_table.rowCount()):
            control = self.fact_table.cellWidget(row, 2)
            verdicts.append({
                "fact": self.fact_table.item(row, 0).text(),
                "verdict": control.currentText() if isinstance(control, QComboBox) else "待审核",
            })
        path = Path(self._run_dir) / "human_review.json"
        try:
            existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            existing[case_id] = {
                **(existing.get(case_id) if isinstance(existing.get(case_id), dict) else {}),
                "behavior_verdict": self.behavior_review.currentText(),
                "required_facts": verdicts,
            }
            path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "保存失败", f"无法保存人工核对结果：{exc}")
            return
        QMessageBox.information(self, "已保存", f"人工核对结果已保存在：\n{path}")

    def _load_case_review(self, case_id: str) -> dict[str, Any]:
        if not self._run_dir:
            return {}
        path = Path(self._run_dir) / "human_review.json"
        try:
            content = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            return {}
        review = content.get(case_id, {}) if isinstance(content, dict) else {}
        return review if isinstance(review, dict) else {}

    def _choose_history_dir(self, field: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择评测运行目录", field.text() or self.output_path.text())
        if path:
            field.setText(path)

    def _refresh_history_defaults(self) -> None:
        if not self._run_dir:
            return
        current = Path(self._run_dir)
        self.history_new_path.setText(str(current))
        siblings = sorted(
            (path.parent for path in current.parent.glob("*/summary.json") if path.parent != current),
            key=lambda path: path.name,
            reverse=True,
        )
        if siblings and not self.history_old_path.text().strip():
            self.history_old_path.setText(str(siblings[0]))

    def _compare_runs(self) -> None:
        try:
            old_dir = Path(self.history_old_path.text().strip())
            new_dir = Path(self.history_new_path.text().strip())
            old_meta = json.loads((old_dir / "run.json").read_text(encoding="utf-8"))
            new_meta = json.loads((new_dir / "run.json").read_text(encoding="utf-8"))
            old_summary = json.loads((old_dir / "summary.json").read_text(encoding="utf-8"))
            new_summary = json.loads((new_dir / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.history_result.setPlainText(f"无法读取两次运行的报告文件：{exc}")
            return
        comparable_fields = ("dataset_sha256", "ai_bundle_id", "mode")
        differences = [
            f"{field} 不同：基线={old_meta.get(field)}；对比={new_meta.get(field)}"
            for field in comparable_fields
            if old_meta.get(field) != new_meta.get(field)
        ]
        old_settings = old_meta.get("settings", {})
        new_settings = new_meta.get("settings", {})
        for field in ("embedding_model", "embedding_revision", "reranker_model", "reranker_revision", "llm_model", "api_style", "limits"):
            if old_settings.get(field) != new_settings.get(field):
                differences.append(f"settings.{field} 不同：基线={old_settings.get(field)}；对比={new_settings.get(field)}")
        lines = [f"基线：{old_meta.get('run_id', old_dir.name)}", f"对比：{new_meta.get('run_id', new_dir.name)}"]
        if differences:
            lines.extend(["", "注意：两次运行配置不同，指标变化不能只归因于代码变更：", *[f"- {item}" for item in differences]])
        lines.extend(["", "阶段指标：基线命中率 → 对比命中率；证据召回率；MRR；指定反例候选数"])
        for stage in STAGE_LABELS:
            old_values = old_summary.get("stages", {}).get(stage, {})
            new_values = new_summary.get("stages", {}).get(stage, {})
            old_hit, new_hit = old_values.get("gold_hit_rate"), new_values.get("gold_hit_rate")
            old_recall, new_recall = old_values.get("mean_evidence_recall"), new_values.get("mean_evidence_recall")
            lines.append(
                f"{STAGE_LABELS[stage]}：{_percent(old_hit)} → {_percent(new_hit)}（{_delta(old_hit, new_hit)}）；"
                f"证据召回 {_percent(old_recall)} → {_percent(new_recall)}（{_delta(old_recall, new_recall)}）；"
                f"MRR {_percent(old_values.get('mean_reciprocal_rank'))} → {_percent(new_values.get('mean_reciprocal_rank'))}；"
                f"反例 {old_values.get('hard_negative_candidate_count', 0)} → {new_values.get('hard_negative_candidate_count', 0)}"
            )
        lines.extend(["", f"行为匹配率：{_percent(old_summary.get('behavior_accuracy'))} → {_percent(new_summary.get('behavior_accuracy'))}"])
        self.history_result.setPlainText("\n".join(lines))

    def _open_report_dir(self) -> None:
        if self._run_dir and Path(self._run_dir).is_dir():
            open_file(Path(self._run_dir))

    @staticmethod
    def _stage_explanation(stage: str) -> str:
        return {
            "lexical": "关键词检索：查看问题分析得到的关键词是否命中正确正文块。",
            "vector": "向量检索：查看语义相近候选及其相似度分数；分数不是答案正确概率。",
            "fusion": "融合：关键词与向量结果按 RRF 排序合并。",
            "deduplicated": "去重：查看相似正文块是否被合并，以及留下哪一个。",
            "rerank": "重排：查看 Cross-Encoder 对问题与候选正文联合打分后的顺序。",
            "constraints": "条件检查：查看范围、阶段、对象等业务条件是否与候选冲突。",
            "evidence": "证据筛选：查看候选通过或未通过证据门槛的原因。",
            "context": "送入上下文：查看实际提供给生成模型的证据块。",
        }.get(stage, "查看该阶段的候选、分数和处理原因。")


def _value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _percent(value: object) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.1f}%"


def _format_rank_history(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "未进入后续阶段"
    labels = {
        "lexical": "关键词",
        "vector": "向量",
        "fusion": "融合",
        "deduplicated": "去重",
        "rerank": "重排",
        "constraints": "条件检查",
        "evidence": "最终证据",
        "context": "上下文",
    }
    return " → ".join(
        f"{labels.get(str(stage), stage)} #{rank}"
        for stage, rank in value
    )


def _delta(old: object, new: object) -> str:
    if old is None or new is None:
        return "—"
    return f"{(float(new) - float(old)) * 100:+.1f} 个百分点"
