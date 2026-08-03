# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, copy_metadata

project_dir = Path.cwd()
skip_ocr = os.environ.get("SKIP_OCR") == "1"

datas = [
    ("local_full_text_search/ui/resources", "local_full_text_search/ui/resources"),
    ("local_full_text_search/ui/styles", "local_full_text_search/ui/styles"),
    ("README.md", "."),
    ("docs/启动与分发说明.md", "."),
    ("使用说明.md", "."),
    ("配置说明.md", "."),
    ("数据库结构说明.md", "."),
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
    "lxml",
    "lxml.etree",
    "psutil",
]

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
