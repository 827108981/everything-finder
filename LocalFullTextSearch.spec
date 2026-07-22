# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_all

project_dir = Path.cwd()

datas = [
    ("local_full_text_search/ui/resources", "local_full_text_search/ui/resources"),
    ("local_full_text_search/ui/styles", "local_full_text_search/ui/styles"),
    ("ocr_models", "ocr_models"),
    ("README.md", "."),
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
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name="本地多格式全文搜索工具",
)
