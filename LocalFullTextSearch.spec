# -*- mode: python ; coding: utf-8 -*-

import os
import json
from datetime import datetime
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, copy_metadata

project_dir = Path.cwd()
skip_ocr = os.environ.get("SKIP_OCR") == "1"
build_timestamp = os.environ.get("LFTS_BUILD_TIMESTAMP") or datetime.now().strftime(
    "%Y%m%d%H%M%S"
)
build_info_path = project_dir / "build" / "build_info.json"
build_info_path.parent.mkdir(parents=True, exist_ok=True)
build_info_path.write_text(
    json.dumps(
        {
            "application_version": "1.6.15",
            "build_timestamp": build_timestamp,
        },
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

datas = [
    ("local_full_text_search/ui/resources", "local_full_text_search/ui/resources"),
    ("local_full_text_search/ui/styles", "local_full_text_search/ui/styles"),
    ("README.md", "."),
    ("docs/启动与分发说明.md", "."),
    ("使用说明.md", "."),
    ("配置说明.md", "."),
    ("数据库结构说明.md", "."),
    (str(build_info_path), "."),
]
binaries = []
hiddenimports = [
    "fitz",
    "docx",
    "openpyxl",
    "pptx",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL.ImageFont",
    "charset_normalizer",
    "watchdog",
    "win32com",
    "pythoncom",
    "win32api",
    "win32job",
    "win32process",
    "win32service",
    "win32serviceutil",
    "servicemanager",
    "win32event",
    "win32pipe",
    "win32file",
    "win32security",
    "ntsecuritycon",
    "pywintypes",
    "lxml",
    "lxml.etree",
    "psutil",
    "keyring.backends.Windows",
    "keyring.backends.Windows.WinVaultKeyring",
    "tokenizers",
    "onnxruntime",
    "tiktoken",
]

# RAG dependencies remain lazy at runtime so ordinary keyword search works
# before an embedding model is downloaded or an API key is configured. The
# frozen distribution still carries their Python modules and ONNX runtime.
for package_name in ("fastembed", "onnxruntime", "keyring", "tiktoken"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports
for distribution_name in ("fastembed", "onnxruntime", "keyring", "tiktoken"):
    datas += copy_metadata(distribution_name)

if not skip_ocr:
    datas += [
        ("ocr_models/manifest.json", "ocr_models"),
        ("ocr_models/PP-OCRv4_mobile_det", "ocr_models/PP-OCRv4_mobile_det"),
        ("ocr_models/PP-OCRv4_mobile_rec", "ocr_models/PP-OCRv4_mobile_rec"),
    ]
    hiddenimports += [
        "paddleocr",
        "paddle",
        "paddlex",
        "cv2",
        "numpy",
        "pandas",
        "yaml",
        "shapely",
        "pyclipper",
    ]
    for package_name in ("paddleocr", "paddlex", "paddle"):
        package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
        datas += package_datas
        binaries += package_binaries
        hiddenimports += package_hiddenimports
    # PaddleX checks OCR extras through importlib.metadata at runtime. The
    # modules alone are insufficient in a frozen app; their dist-info folders
    # must be present for is_extra_available("ocr-core") to succeed.
    for distribution_name in (
        "imagesize",
        "opencv-contrib-python",
        "pyclipper",
        "pypdfium2",
        "python-bidi",
        "shapely",
    ):
        datas += copy_metadata(distribution_name)

a = Analysis(
    ["app.py"],
    pathex=[str(project_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# Codex's bundled Poppler runtime may be present on PATH while packaging.
# PyInstaller can then resolve Qt6Core's Windows ICU dependency to Poppler's
# incompatible private ICU build and place it at the application root. Qt
# subsequently fails to import QtWidgets. Windows supplies the forwarding ICU
# DLL expected by PySide6, so exclude only these accidentally collected files.
_external_icu_names = {"icuuc.dll", "icudt78.dll"}
a.binaries = [
    entry
    for entry in a.binaries
    if Path(entry[0]).name.lower() not in _external_icu_names
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="本地多格式全文搜索工具",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="本地多格式全文搜索工具",
)
