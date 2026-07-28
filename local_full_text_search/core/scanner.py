from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Iterable

from local_full_text_search.config.constants import DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILE_PATTERNS
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken


def iter_files(
    root_path: Path,
    *,
    include_subfolders: bool,
    settings: AppSettings,
    cancel_token: CancelToken,
) -> Iterable[Path]:
    excluded_dirs = set(settings.excluded_dirs or DEFAULT_EXCLUDED_DIRS)
    excluded_patterns = tuple(settings.excluded_file_patterns or DEFAULT_EXCLUDED_FILE_PATTERNS)
    if include_subfolders:
        for current, dirs, files in os.walk(root_path, onerror=_raise_walk_error):
            cancel_token.wait_if_paused()
            cancel_token.throw_if_cancelled()
            dirs[:] = [name for name in dirs if name not in excluded_dirs]
            for filename in files:
                cancel_token.wait_if_paused()
                cancel_token.throw_if_cancelled()
                path = Path(current) / filename
                if _is_excluded_file(path, excluded_patterns):
                    continue
                if not settings.include_hidden_files and _is_hidden(path):
                    continue
                yield path
    else:
        for path in root_path.iterdir():
            cancel_token.wait_if_paused()
            cancel_token.throw_if_cancelled()
            if not path.is_file():
                continue
            if _is_excluded_file(path, excluded_patterns):
                continue
            if not settings.include_hidden_files and _is_hidden(path):
                continue
            yield path


def _is_excluded_file(path: Path, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)


def _is_hidden(path: Path) -> bool:
    try:
        if path.name.startswith("."):
            return True
        if os.name == "nt":
            import ctypes

            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
            return attrs != -1 and bool(attrs & 0x2)
    except Exception:
        return False
    return False


def _raise_walk_error(error: OSError) -> None:
    # An incomplete walk must not make the deletion pass remove previously indexed files.
    raise error
