from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_full_text_search.core.hang_validation import (
    run_semantic_progress_validation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("semantic_progress_validation_result.json"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=0.5)
    args = parser.parse_args()
    report = run_semantic_progress_validation(
        args.output,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
