from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "LocalFullTextSearch"
APP_DISPLAY_NAME = "本地多格式全文搜索工具"

DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".xlsm",
    ".ppt",
    ".pptx",
    ".txt",
    ".log",
    ".csv",
    ".md",
    ".json",
    ".xml",
    ".ini",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

ARCHIVE_EXTENSIONS = {".zip"}

MEDIA_METADATA_EXTENSIONS = {".mp4"}

LEGACY_OFFICE_EXTENSIONS = {".doc", ".xls", ".ppt"}

SUPPORTED_EXTENSIONS = DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS | ARCHIVE_EXTENSIONS | MEDIA_METADATA_EXTENSIONS

TEXT_EXTENSIONS = {
    ".txt",
    ".log",
    ".csv",
    ".md",
    ".json",
    ".xml",
    ".ini",
}

DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".svn",
    "node_modules",
    "__pycache__",
    "$RECYCLE.BIN",
    "System Volume Information",
}

DEFAULT_EXCLUDED_FILE_PATTERNS = (
    "~$*.docx",
    "~$*.xlsx",
    "~$*.pptx",
    "*.tmp",
    "*.temp",
    "*.part",
    "*.crdownload",
)

FILE_TYPE_GROUPS = {
    "全部": set(),
    "PDF": {".pdf"},
    "Word": {".doc", ".docx"},
    "Excel": {".xls", ".xlsx", ".xlsm"},
    "PowerPoint": {".ppt", ".pptx"},
    "文本/日志": TEXT_EXTENSIONS,
    "图片": IMAGE_EXTENSIONS,
    "压缩包": ARCHIVE_EXTENSIONS,
    "其他": MEDIA_METADATA_EXTENSIONS,
}


def app_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / APP_NAME
    return Path.home() / f".{APP_NAME}"


APP_DATA_DIR = app_data_dir()
DATA_DIR = APP_DATA_DIR / "data"
LOG_DIR = APP_DATA_DIR / "logs"
CONFIG_DIR = APP_DATA_DIR / "config"
CACHE_DIR = APP_DATA_DIR / "cache"
TEMP_DIR = APP_DATA_DIR / "temp"
OCR_CACHE_DIR = CACHE_DIR / "ocr_cache"
DB_PATH = DATA_DIR / "search_index.db"
SETTINGS_PATH = CONFIG_DIR / "settings.json"


def bundled_resource_dir(name: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")) / name
    return Path(__file__).resolve().parents[2] / name


OCR_MODELS_DIR = bundled_resource_dir("ocr_models")

PARSER_VERSION = "3"

PARSER_VERSIONS = {
    "docx": "2",
    "docx_stream": "1",
    "pptx": "2",
    "pptx_stream": "1",
    "xlsx": "2",
    "xlsx_stream": "1",
    "pdf": "3",
    "image_ocr": "3",
    "zip": "3",
    "legacy_office": "2",
    "text": "2",
    "metadata": "2",
}
