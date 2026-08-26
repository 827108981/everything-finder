from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_window(window: object, application: object, output: Path) -> None:
    application.processEvents()
    pixmap = window.grab()
    if pixmap.width() != 1280 or pixmap.height() != 800:
        raise RuntimeError(
            f"Unexpected guide screenshot size: {pixmap.width()}x{pixmap.height()}"
        )
    if not pixmap.save(str(output), "PNG"):
        raise RuntimeError(f"Unable to save guide screenshot: {output}")
    with Image.open(output) as image:
        colors = image.convert("RGB").resize((64, 40)).getcolors(maxcolors=4096)
        if colors is None or len(colors) < 12:
            raise RuntimeError(f"Guide screenshot appears blank: {output}")


def crop_asset(source: Path, target: Path, box: tuple[int, int, int, int]) -> None:
    with Image.open(source) as image:
        cropped = image.crop(box)
        cropped.save(target, "PNG", optimize=True)


def build_assets(output: Path) -> list[Path]:
    os.environ.setdefault("QT_QPA_PLATFORM", "windows" if os.name == "nt" else "offscreen")
    os.environ.setdefault("QT_SCALE_FACTOR", "1")
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication, QPushButton

    from app import apply_light_theme
    from local_full_text_search.config.defaults import AppSettings
    from local_full_text_search.core.database import DatabaseManager
    from local_full_text_search.core.search_engine import SearchPage
    from local_full_text_search.models.search_result import SearchHit, SearchResult
    from local_full_text_search.services.settings_service import SettingsService
    from local_full_text_search.ui.main_window import MainWindow
    from local_full_text_search.version import __version__

    output.mkdir(parents=True, exist_ok=True)
    application = QApplication.instance() or QApplication(
        ["LocalFullTextSearch", "--generate-user-guide-assets"]
    )
    available_fonts = set(QFontDatabase.families())
    chinese_font = next(
        (
            family
            for family in ("Microsoft YaHei UI", "Microsoft YaHei", "DengXian")
            if family in available_fonts
        ),
        "",
    )
    if os.name == "nt" and not chinese_font:
        raise RuntimeError("A Windows Chinese UI font is required for guide screenshots")
    if chinese_font:
        application.setFont(QFont(chinese_font, 9))
    apply_light_theme(application)

    saved: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="lfts_user_guide_") as tmp:
        base = Path(tmp)
        settings_service = SettingsService(base / "settings.json")
        settings = AppSettings(enable_ocr=True, monitor_file_changes=True)
        settings_service.save(settings)
        database = DatabaseManager(base / "guide.db")
        database.initialize()
        window = MainWindow(database, settings, settings_service)
        window.resize(1280, 800)
        window.show()

        query_text = "拔掉 3 个传感器"
        hit = SearchHit(
            block_id=6,
            location_text="正文 · 第 6 页",
            context=(
                "第 6 页 操作前请拔掉 3 个传感器，并标记插头。"
                "完成后按拆卸步骤移除组件。"
            ),
            hit_count=1,
            source_type="native_text",
            page_number=6,
        )
        result = SearchResult(
            file_id=6,
            block_id=6,
            file_path=r"D:\设备资料\免疫资料\第一抓杯手手指组件拆卸方法.pdf",
            filename="第一抓杯手手指组件拆卸方法.pdf",
            extension=".pdf",
            size_bytes=3_145_728,
            modified_time=1_786_052_400,
            location_text=hit.location_text,
            context=hit.context,
            hit_count=1,
            source_type=hit.source_type,
            parse_status="success",
            score=1.0,
            page_number=6,
            matches=[hit],
        )
        page = SearchPage(
            results=[result],
            total_candidates=16,
            total_confirmed=1,
            elapsed_ms=87,
            page=1,
            page_size=20,
            available_results=1,
            truncated=False,
        )
        window.search_page.set_stats(
            {"files": 309, "indexed": 309, "failed": 0},
            has_roots=True,
        )
        window.search_page.set_index_ready(
            True,
            {
                "ready": True,
                "blocking_files": 0,
                "metadata_only_complete_files": 22,
                "not_ready_reasons": [],
            },
        )
        window.search_page.search_box.input.setText(query_text)
        window.search_page.set_results(page)
        window.show_preview(result)
        window.top_bar.set_index_status("当前范围完成 309/309 · 排除视频 22")
        search_path = output / "search_results.png"
        save_window(window, application, search_path)
        saved.append(search_path)

        root_row = {
            "id": 1,
            "path": r"D:\设备资料\免疫资料",
            "enabled": 1,
            "last_scan_at": "2026-08-14 09:30",
        }
        root_stats = {
            ".pdf": 65,
            ".docx": 121,
            ".doc": 31,
            ".xlsx": 27,
            ".xls": 7,
            ".pptx": 39,
            ".ppt": 12,
            ".jpg": 6,
            ".txt": 1,
        }
        window.switch_page("index")
        window.index_page.set_roots([root_row], {1: root_stats})
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        for button in window.index_page.list_host.findChildren(QPushButton):
            if button.text() == "添加搜索范围":
                button.hide()
        progress = {
            "stage": "indexing",
            "phase_label": "正在解析并写入索引",
            "indexed": 209,
            "scanned": 309,
            "completed_files": 209,
            "total_files": 309,
            "failed": 0,
            "current_file": r"D:\设备资料\免疫资料\9000i\微信图片_设备说明.jpg",
            "eta_seconds": 660,
            "eta_ready": True,
            "eta_confidence": 0.84,
            "active_queue": "ocr",
            "active_elapsed_seconds": 252,
            "active_file_count": 2,
            "active_phase": "ocr_recognize_original_regions",
            "active_completed_units": 128,
            "active_total_units": 240,
            "no_progress_seconds": 7,
            "retry_count": 0,
            "excluded_video": 22,
        }
        window.on_scan_progress(progress)
        running_path = output / "index_running.png"
        save_window(window, application, running_path)
        saved.append(running_path)

        window.index_page.set_pause_state("paused", "普通模式 · 已暂停")
        window.top_bar.set_index_status("普通模式 · 已暂停 209 / 309", is_pending=True)
        paused_path = output / "index_paused.png"
        save_window(window, application, paused_path)
        saved.append(paused_path)

        window.switch_page("failed")
        window.failed_page.set_rows(
            [
                {
                    "id": 31,
                    "path": r"D:\设备资料\免疫资料\加密维护记录.doc",
                    "filename": "加密维护记录.doc",
                    "extension": ".doc",
                    "parse_status": "failed",
                    "parse_error_message": "文件受密码保护，无法读取正文",
                },
                {
                    "id": 32,
                    "path": r"D:\设备资料\免疫资料\损坏扫描件.pdf",
                    "filename": "损坏扫描件.pdf",
                    "extension": ".pdf",
                    "parse_status": "failed",
                    "parse_error_message": "PDF 结构损坏，最后一次恢复重试失败",
                },
            ]
        )
        window.failed_page.set_readiness(
            {
                "ready": False,
                "repairable": True,
                "not_ready_reasons": ["blocking_files"],
            }
        )
        window.top_bar.set_index_status("索引未开放 · 阻断项 2", is_error=True)
        failed_path = output / "failed_files.png"
        save_window(window, application, failed_path)
        saved.append(failed_path)

        window.switch_page("settings")
        settings_page = window.settings_page
        settings_page.database_path_value.setText(r"D:\全文索引\search_index.db")
        settings_page.database_path_value.setToolTip(r"D:\全文索引\search_index.db")
        settings_page.database_size_value.setText("346.7 MB")
        settings_page.database_schema_value.setText("schema v8")
        settings_page.database_current_space_value.setText("D: · 可用 612.4 GB")
        settings_page.database_target.setText(r"E:\本地搜索数据库")
        settings_page.database_target_space_value.setText("E: · 可用 824.1 GB")
        settings_page.database_migration_status.setText("数据库未迁移")
        settings_page._refresh_database_actions()
        window.top_bar.set_index_status("当前范围完成 309/309 · 排除视频 22")
        settings_path = output / "settings_database.png"
        save_window(window, application, settings_path)
        saved.append(settings_path)

        window.close()
        application.processEvents()

    crops = (
        ("search_results.png", "search_controls.png", (185, 76, 918, 260)),
        ("index_running.png", "index_status.png", (182, 78, 1270, 258)),
        ("settings_database.png", "database_settings_detail.png", (180, 72, 1270, 520)),
    )
    for source_name, target_name, box in crops:
        target = output / target_name
        crop_asset(output / source_name, target, box)
        saved.append(target)

    manifest = {
        "schema_version": 1,
        "application_version": __version__,
        "render_size": [1280, 800],
        "font_family": chinese_font,
        "data_notice": "当前版本真实界面组件，内容为演示数据",
        "files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in saved
        ],
    }
    manifest_path = output / "assets-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render real current-version UI screenshots for the PDF user guide"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "assets" / "user-guide",
    )
    args = parser.parse_args()
    saved = build_assets(args.output.resolve())
    print(f"USER_GUIDE_ASSETS_OK files={len(saved)} output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
