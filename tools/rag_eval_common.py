"""Shared, behavior-neutral helpers for local RAG experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path: str | Path, *, skip_blank: bool = False) -> list[Any]:
    source = Path(path).read_text(encoding="utf-8").splitlines()
    if skip_blank:
        source = [line for line in source if line.strip()]
    return [json.loads(line) for line in source]


def write(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def redact(value: object, secret: str, *, limit: int | None = None) -> str:
    result = str(value).replace(secret, "[REDACTED]") if secret else str(value)
    return result[:limit] if limit is not None else result


def pair_order(index: int, repeat: int) -> str:
    first = "A" if (index + repeat) % 2 else "B"
    return first + ("B" if first == "A" else "A")
