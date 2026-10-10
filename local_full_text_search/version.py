from __future__ import annotations

import json
import os
import sys
from pathlib import Path

__version__ = "1.6.15"


def _load_build_timestamp() -> str:
    """Read a timestamp baked into a frozen package, not its runtime env."""

    configured = os.environ.get("LFTS_BUILD_TIMESTAMP")
    if configured:
        return str(configured)
    if bool(getattr(sys, "frozen", False)):
        metadata_path = Path(getattr(sys, "_MEIPASS", Path(sys.executable))) / "build_info.json"
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            value = str(payload.get("build_timestamp") or "").strip()
            if value:
                return value
        except (OSError, TypeError, ValueError):
            pass
    # A source checkout has no release timestamp.  Showing "dev" is safer
    # than displaying a stale timestamp copied from an old installer.
    return "dev"


__build_timestamp__ = _load_build_timestamp()
