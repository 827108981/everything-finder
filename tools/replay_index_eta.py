from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_full_text_search.core.eta_replay import load_replay_events, replay_eta


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline replay of single-value indexing ETA events."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        events = load_replay_events(payload)
        report = replay_eta(events).to_dict()
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0
    except Exception as exc:
        print(f"ETA_REPLAY_FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
