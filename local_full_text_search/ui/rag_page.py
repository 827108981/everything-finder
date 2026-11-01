from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from local_full_text_search.ui.rag_diagnostic_dialog import RagDiagnosticDialog


class RagPage(QWidget):
    download_models_requested = Signal()
    build_index_requested = Signal()
    ask_requested = Signal(str)
    stop_requested = Signal()
    source_open_requested = Signal(str, object)

    def __init__(self) -> None:
        super().__init__()
        self._rag_enabled = False
        self._busy = False
        self._models_ready = False
        self._index_ready = False
        self._embedding_repo_id = ""
        self.setObjectName("Page")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        heading = QLabel("资料智能问答")
        heading.setObjectName("PageTitle")
        intro = QLabel(
            "首次使用时下载本地向量与重排模型，再基于已完成的全文索引构建 AI 索引。"
            "后续启动会自动复用磁盘中的模型和已发布 AI 索引，无需重复下载或构建。"
            "提问时，问题、文档和引用来源在本机检索；答案请求会发送到设置页填写的 LLM API。"
        )
        intro.setWordWrap(True)
        intro.setObjectName("MutedText")
        layout.addWidget(heading)
        layout.addWidget(intro)

        self.embedding_model_label = QLabel("本地向量模型：等待读取设置")
        self.embedding_model_label.setObjectName("InlineStatus")
        self.embedding_model_label.setWordWrap(True)
        layout.addWidget(self.embedding_model_label)

        controls = QHBoxLayout()
        self.download_button = QPushButton("下载 / 校验模型")
        self.build_button = QPushButton("构建 / 更新 AI 索引")
        self.stop_button = QPushButton("停止当前任务")
        self.stop_button.setEnabled(False)
        controls.addWidget(self.download_button)
        controls.addWidget(self.build_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.status_label = QLabel("AI 索引尚未构建")
        self.status_label.setObjectName("InlineStatus")
        self.status_label.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)

        self.question = QTextEdit()
        self.question.setPlaceholderText("例如：索引任务中断后，程序如何恢复？")
        self.question.setFixedHeight(92)
        layout.addWidget(self.question)
        ask_row = QHBoxLayout()
        self.diagnostics_button = QPushButton("查看检索诊断")
        self.ask_button = QPushButton("提问")
        self.ask_button.setObjectName("PrimaryButton")
        ask_row.addWidget(self.diagnostics_button)
        ask_row.addStretch(1)
        ask_row.addWidget(self.ask_button)
        layout.addLayout(ask_row)

        answer_card = QFrame()
        answer_card.setObjectName("SettingsCard")
        answer_layout = QVBoxLayout(answer_card)
        answer_layout.setContentsMargins(16, 14, 16, 14)
        answer_layout.setSpacing(8)
        answer_layout.addWidget(QLabel("答案"))
        self.answer = QTextEdit()
        self.answer.setReadOnly(True)
        self.answer.setPlaceholderText("答案和引用将在这里显示。")
        self.answer.setMinimumHeight(210)
        answer_layout.addWidget(self.answer)
        answer_layout.addWidget(QLabel("引用来源（双击打开到对应位置）"))
        self.sources = QListWidget()
        self.sources.setMaximumHeight(170)
        answer_layout.addWidget(self.sources)
        layout.addWidget(answer_card, 1)

        self.diagnostics_dialog = RagDiagnosticDialog(self)

        self.download_button.clicked.connect(self.download_models_requested)
        self.build_button.clicked.connect(self.build_index_requested)
        self.ask_button.clicked.connect(self._ask)
        self.diagnostics_button.clicked.connect(self.diagnostics_dialog.show)
        self.stop_button.clicked.connect(self.stop_requested)
        self.sources.itemDoubleClicked.connect(self._open_source)

    def set_rag_enabled(self, enabled: bool) -> None:
        self._rag_enabled = bool(enabled)
        self._update_action_states()
        if not self._rag_enabled:
            self.status_label.setText("资料智能问答未启用，可在设置页开启")

    def set_embedding_model(
        self, repo_id: str, label: str, dimension: int = 0, *, invalidate_index: bool = False,
    ) -> None:
        changed = bool(self._embedding_repo_id and self._embedding_repo_id != str(repo_id))
        self._embedding_repo_id = str(repo_id)
        dimension_text = f" · {int(dimension)} 维" if dimension else ""
        self.embedding_model_label.setText(
            f"当前配置：{label}{dimension_text}\n{self._embedding_repo_id}"
        )
        if changed and invalidate_index:
            self._index_ready = False
            self.status_label.setText(
                "向量模型设置已改变，请先下载 / 校验模型，再重新构建完整 AI 索引"
            )
        self._update_action_states()

    def _update_action_states(self) -> None:
        self.download_button.setEnabled(not self._busy)
        self.build_button.setEnabled(not self._busy)
        self.ask_button.setEnabled(
            not self._busy and self._rag_enabled and self._index_ready
        )
        self.stop_button.setEnabled(self._busy)

    def _ask(self) -> None:
        question = self.question.toPlainText().strip()
        if question:
            self.ask_requested.emit(question)

    def _open_source(self, item: QListWidgetItem) -> None:
        payload = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict):
            return
        payload = dict(payload)
        path = str(payload.pop("file_path", ""))
        if path:
            self.source_open_requested.emit(path, payload)

    def set_busy(self, busy: bool, label: str = "正在处理") -> None:
        self._busy = bool(busy)
        self._update_action_states()
        self.progress.setVisible(busy)
        if busy:
            self.progress.setRange(0, 0)
            self.progress.setFormat("正在准备…")
            self.status_label.setText(label)

    def show_progress(self, payload: object) -> None:
        if isinstance(payload, dict):
            self.diagnostics_dialog.update_progress(payload)
            message = str(payload.get("message") or payload.get("model") or payload.get("stage") or "正在处理")
            count = payload.get("chunk_count")
            total = payload.get("total_chunk_count")
            if count is not None and total is not None and int(total) > 0:
                current = max(0, min(int(count), int(total)))
                total_count = int(total)
                percent = float(payload.get("percent") or current * 100 / total_count)
                self.progress.setRange(0, 1000)
                self.progress.setValue(max(0, min(1000, int(round(percent * 10)))))
                self.progress.setFormat(f"{current:,} / {total_count:,}（{percent:.1f}%）")
                message += f" · {current:,} / {total_count:,} 个内容块（{percent:.1f}%）"
                eta = payload.get("eta_seconds")
                if eta is not None and int(eta) >= 0:
                    message += f" · 预计剩余 {self._format_duration(int(eta))}"
                resumed = int(payload.get("resumed_chunk_count") or 0)
                if resumed:
                    message += f" · 已恢复 {resumed:,} 个"
            elif count is not None:
                message += f" · 已处理 {int(count):,} 个内容块"
            else:
                parent_count = payload.get("processed_parent_chunks")
                parent_total = payload.get("total_parent_chunks")
                if parent_count is not None and parent_total is not None and int(parent_total) > 0:
                    current = max(0, min(int(parent_count), int(parent_total)))
                    total_count = int(parent_total)
                    percent = float(payload.get("percent") or current * 100 / total_count)
                    self.progress.setRange(0, 1000)
                    self.progress.setValue(max(0, min(1000, int(round(percent * 10)))))
                    self.progress.setFormat(
                        f"细粒度索引 {current:,} / {total_count:,}（{percent:.1f}%）"
                    )
                    message += (
                        f" · 父内容块 {current:,} / {total_count:,}（{percent:.1f}%）"
                        f" · 已生成 {int(payload.get('unit_count') or 0):,} 个细粒度向量单元"
                    )
                    eta = payload.get("eta_seconds")
                    if eta is not None and int(eta) >= 0:
                        message += f" · 预计剩余 {self._format_duration(int(eta))}"
            self.status_label.setText(message)

    @staticmethod
    def _format_duration(seconds: int) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours} 小时 {minutes} 分钟"
        if minutes:
            return f"{minutes} 分钟 {secs} 秒"
        return f"{secs} 秒"

    def start_diagnostics(self, request_id: str, question: str) -> None:
        self.diagnostics_dialog.start_query(request_id, question)

    def finish_diagnostics(self, status: str) -> None:
        self.diagnostics_dialog.mark_finished(status)

    def show_build_summary(self, summary: object) -> None:
        count = int(getattr(summary, "chunk_count", 0) or 0)
        generation = int(getattr(summary, "source_generation", 0) or 0)
        self._index_ready = True
        self._models_ready = True
        self._update_action_states()
        unit_count = int(getattr(summary, "unit_count", 0) or 0)
        count_text = (
            f"{count:,} 个父内容块 · {unit_count:,} 个细粒度向量"
            if unit_count else f"{count:,} 个内容块"
        )
        if bool(getattr(summary, "unchanged", False)):
            self.status_label.setText(
                f"资料未变化，沿用已有 AI 索引 · 来源 generation {generation} · {count_text}"
            )
            return
        self.status_label.setText(
            f"AI 索引已就绪 · 来源 generation {generation} · {count_text}"
        )

    def show_local_state(self, state: object) -> None:
        if not isinstance(state, dict):
            self.status_label.setText("未能读取本地 AI 资源状态")
            return
        models_ready = bool(state.get("models_ready"))
        index_ready = bool(state.get("index_ready"))
        self._models_ready = models_ready
        self._index_ready = index_ready
        self._update_action_states()
        configured_model = str(
            state.get("configured_embedding_model") or self._embedding_repo_id or "BGE"
        )
        index_model = str(state.get("index_embedding_model") or "")
        if models_ready and index_ready:
            message = (
                f"已复用本地 {configured_model} 模型和已发布 AI 索引"
                f" · {int(state.get('chunk_count') or 0):,} 个内容块"
                f" · {int(state.get('vector_count') or 0):,} 个 Chroma 向量"
            )
            if state.get("using_previous_index"):
                message += " · 全文索引已有更新，当前继续使用旧 AI 索引"
            self.status_label.setText(message)
            return
        messages = []
        if not models_ready:
            messages.append(
                f"本地 {configured_model} 模型不完整，请点击“下载 / 校验模型”"
            )
        if not index_ready:
            detail = str(state.get("index_error") or "尚未构建 AI 索引")
            if index_model and index_model != configured_model:
                detail += f"（当前索引：{index_model}）"
            messages.append(f"本地 AI 索引不可用：{detail}")
        self.status_label.setText("；".join(messages))

    def show_answer(self, answer: object) -> None:
        insufficient = bool(getattr(answer, "insufficient_evidence", False))
        answer_status = str(
            getattr(
                getattr(answer, "status", ""),
                "value",
                getattr(answer, "status", ""),
            )
        )
        model_called = bool(getattr(answer, "provider_name", ""))
        # 最终答案也属于诊断链路的一部分：只有写回诊断窗口，测试人员才能
        # 对照“召回 → 重排 → LLM 请求 → 最终结果”定位偏差发生在哪一层。
        self.diagnostics_dialog.set_final_answer(answer)
        if answer_status == "needs_clarification":
            self.finish_diagnostics("需要补充信息")
        elif insufficient:
            self.finish_diagnostics(
                "LLM 判断证据不足" if model_called else "证据不足，未调用 LLM"
            )
        else:
            self.finish_diagnostics("问答已完成")
        self.answer.setPlainText(str(getattr(answer, "answer_text", "")))
        self.sources.clear()
        sources = tuple(getattr(answer, "sources", ()))
        source_ids = tuple(getattr(answer, "source_ids", ()))
        for index, source in enumerate(sources, start=1):
            location = str(getattr(source, "location_text", "") or "")
            filename = str(getattr(source, "filename", Path(str(getattr(source, "file_path", ""))).name))
            source_id = source_ids[index - 1] if index <= len(source_ids) else f"S{index}"
            item = QListWidgetItem(f"[{source_id}] {filename}  ·  {location}")
            item.setData(
                Qt.ItemDataRole.UserRole,
                {
                    "file_path": str(getattr(source, "file_path", "")),
                    "location_text": location,
                    "page_number": getattr(source, "page_number", None),
                    "slide_number": getattr(source, "slide_number", None),
                    "sheet_name": getattr(source, "sheet_name", None),
                    "cell_start": getattr(source, "cell_start", None),
                    "cell_end": getattr(source, "cell_end", None),
                    "line_start": getattr(source, "line_start", None),
                    "line_end": getattr(source, "line_end", None),
                },
            )
            self.sources.addItem(item)
        if answer_status == "needs_clarification":
            self.status_label.setText("请补充问题中的场景信息后重新提问")
        elif insufficient:
            self.status_label.setText(
                "模型判断证据不足" if model_called else "检索证据不足，未调用 LLM"
            )
        else:
            model = str(getattr(answer, "model_name", "") or "")
            self.status_label.setText(f"回答完成 · {model}" if model else "回答完成")
        warnings = tuple(getattr(answer, "warnings", ()))
        stale_warning = next((item for item in warnings if "旧 AI 索引" in item), "")
        if stale_warning:
            self.status_label.setText(f"{self.status_label.text()} · {stale_warning}")

    def show_failure(self, message: str) -> None:
        self.status_label.setText(f"任务失败：{message}")

    def show_cancelled(self) -> None:
        self.status_label.setText("任务已停止")

    def show_task_finished(self) -> None:
        self.set_busy(False)
