from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication

from local_full_text_search.core.open_location import (
    _coerce_location,
    _extract_excel_location,
    _extract_line_number,
    _extract_page_number,
    _extract_slide_number,
    _open_pdf_page,
    _open_powerpoint_location,
)
from local_full_text_search.models.search_result import SearchHit, SearchResult
from local_full_text_search.ui.preview_panel import PreviewPanel
from local_full_text_search.ui.result_view import ResultCard


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


class OpenLocationTests(unittest.TestCase):
    def test_extracts_supported_index_locations(self) -> None:
        self.assertEqual(_extract_page_number("正文 · 第 9 页"), 9)
        self.assertEqual(_extract_slide_number("第 12 张幻灯片；备注"), 12)
        self.assertEqual(_extract_line_number("第 501-1000 行"), 501)
        self.assertEqual(
            _extract_excel_location("Sheet：维修记录；第 27 行"),
            ("维修记录", 27),
        )

    def test_pdf_browser_receives_page_fragment(self) -> None:
        with (
            patch(
                "local_full_text_search.core.open_location._find_executable",
                side_effect=[None, None, None, r"C:\Program Files\Microsoft\Edge\msedge.exe"],
            ),
            patch("local_full_text_search.core.open_location.subprocess.Popen") as popen,
        ):
            message = _open_pdf_page(Path(r"C:\docs\manual.pdf"), 9)

        command = popen.call_args.args[0]
        self.assertEqual(command[1], "--new-window")
        self.assertTrue(command[2].endswith("manual.pdf#page=9"))
        self.assertIn("第 9 页", str(message))

    def test_structured_location_is_preferred_over_display_text(self) -> None:
        hit = SearchHit(
            block_id=46,
            location_text="第 1 张幻灯片",
            context="主真空",
            hit_count=1,
            source_type="native_text",
            slide_number=46,
            page_number=9,
            sheet_name="维修记录",
            cell_start="B27",
            cell_end="D27",
            line_start=501,
            line_end=1000,
        )

        location = _coerce_location(hit)

        self.assertEqual(location.slide_number, 46)
        self.assertEqual(location.page_number, 9)
        self.assertEqual(location.sheet_name, "维修记录")
        self.assertEqual(location.cell_start, "B27")
        self.assertEqual(location.line_start, 501)

    def test_powerpoint_uses_active_dispatch_then_dispatchex_and_wps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "slides.pptx"
            target.write_bytes(b"pptx")
            pythoncom = MagicMock()
            client = MagicMock()
            app, presentation, window = _powerpoint_mocks(46)

            def get_active(prog_id: str) -> object:
                if prog_id == "KWPP.Application":
                    return app
                raise RuntimeError("no active Microsoft PowerPoint")

            client.GetActiveObject.side_effect = get_active
            client.Dispatch.side_effect = RuntimeError("dispatch unavailable")
            client.DispatchEx.side_effect = RuntimeError("dispatchex unavailable")

            with patch(
                "local_full_text_search.core.open_location._office_com",
                return_value=(pythoncom, client),
            ):
                message = _open_powerpoint_location(
                    target,
                    SearchHit(
                        block_id=1,
                        location_text="第 1 张幻灯片",
                        context="hit",
                        hit_count=1,
                        source_type="native_text",
                        slide_number=46,
                    ),
                )

        self.assertEqual(message, "已定位到 PowerPoint 第 46 张幻灯片")
        self.assertEqual(client.GetActiveObject.call_args_list[0].args, ("PowerPoint.Application",))
        self.assertEqual(client.Dispatch.call_args_list[0].args, ("PowerPoint.Application",))
        self.assertEqual(client.DispatchEx.call_args_list[0].args, ("PowerPoint.Application",))
        self.assertEqual(client.GetActiveObject.call_args_list[-1].args, ("KWPP.Application",))
        app.Presentations.Open.assert_called_once()
        window.View.GotoSlide.assert_called_once_with(46)
        pythoncom.CoUninitialize.assert_called_once()

    def test_powerpoint_failure_opens_file_without_reporting_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "slides.pptx"
            target.write_bytes(b"pptx")
            pythoncom = MagicMock()
            client = MagicMock()
            client.GetActiveObject.side_effect = RuntimeError("no active instance")
            client.Dispatch.side_effect = RuntimeError("dispatch unavailable")
            client.DispatchEx.side_effect = RuntimeError("dispatchex unavailable")

            with (
                patch(
                    "local_full_text_search.core.open_location._office_com",
                    return_value=(pythoncom, client),
                ),
                patch("local_full_text_search.core.open_location.open_file") as open_file,
            ):
                message = _open_powerpoint_location(
                    target,
                    SearchHit(
                        block_id=1,
                        location_text="第 46 张幻灯片",
                        context="hit",
                        hit_count=1,
                        source_type="native_text",
                        slide_number=46,
                    ),
                )

        open_file.assert_called_once_with(target)
        self.assertIn("已打开文件", str(message))
        self.assertIn("未能定位到第 46 张幻灯片", str(message))
        self.assertNotIn("已定位到 PowerPoint", str(message))

    def test_left_and_right_location_links_emit_same_structured_hit(self) -> None:
        _app()
        hit = SearchHit(
            block_id=46,
            location_text="第 46 张幻灯片",
            context="主真空",
            hit_count=1,
            source_type="native_text",
            slide_number=46,
        )
        result = SearchResult(
            file_id=1,
            block_id=46,
            file_path=r"E:\slides.pptx",
            filename="slides.pptx",
            extension=".pptx",
            size_bytes=1024,
            modified_time=0,
            location_text=hit.location_text,
            context=hit.context,
            hit_count=1,
            source_type=hit.source_type,
            parse_status="success",
            score=1.0,
            slide_number=46,
            matches=[hit],
        )
        left: list[tuple[str, object]] = []
        right: list[tuple[str, object]] = []
        card = ResultCard(result, "主真空")
        preview = PreviewPanel()
        card.open_location_requested.connect(lambda path, value: left.append((path, value)))
        preview.open_location_requested.connect(lambda path, value: right.append((path, value)))

        left_button = card.location_layout.itemAt(0).widget()
        assert left_button is not None
        left_button.click()
        preview.show_result(result, "主真空")
        preview._handle_link(QUrl("hit:0"))

        self.assertEqual(left[0][0], result.file_path)
        self.assertEqual(right[0][0], result.file_path)
        self.assertIs(left[0][1], hit)
        self.assertIs(right[0][1], hit)
        self.assertEqual(getattr(left[0][1], "slide_number"), 46)
        self.assertEqual(getattr(right[0][1], "slide_number"), 46)


def _powerpoint_mocks(slide_number: int) -> tuple[MagicMock, MagicMock, MagicMock]:
    app = MagicMock()
    presentation = MagicMock()
    window = MagicMock()
    app.Presentations.Count = 0
    app.Presentations.Open.return_value = presentation
    presentation.Windows.Count = 1
    presentation.Windows.return_value = window
    window.View.Slide.SlideIndex = slide_number
    return app, presentation, window


if __name__ == "__main__":
    unittest.main()
