from __future__ import annotations

import os
import subprocess
from pathlib import Path


def open_file(path: str | Path) -> None:
    target = _physical_target(path)
    if not target.exists():
        raise FileNotFoundError(str(target))
    if os.name == "nt":
        os.startfile(str(target))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(target)])


def open_parent_folder(path: str | Path) -> None:
    target = _physical_target(path)
    parent = target.parent if target.suffix else target
    if not parent.exists():
        raise FileNotFoundError(str(parent))
    if os.name == "nt":
        subprocess.Popen(["explorer", "/select,", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(parent)])


def _physical_target(path: str | Path) -> Path:
    path_text = str(path)
    if " > " in path_text:
        path_text = path_text.split(" > ", 1)[0]
    return Path(path_text)
