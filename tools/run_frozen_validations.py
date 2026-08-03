from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


VALIDATIONS = (
    "--self-test",
    "--validate-core",
    "--validate-process-pool",
    "--validate-schema-v6",
    "--validate-schema-v8",
    "--validate-manual-exclusion",
    "--validate-failure-demo",
    "--validate-database-lock",
    "--validate-shutdown",
    "--validate-checkpoint-timeout",
    "--validate-hang-recovery",
    "--validate-semantic-progress",
    "--validate-pdf-page-pipeline",
    "--validate-ocr-adaptive-v2",
    "--validate-ocr-backend",
    "--validate-index-status-layout",
    "--validate-single-eta",
    "--validate-safe-pause",
    "--validate-paused-mode-switch",
    "--validate-ui",
)

PARAMETERIZED_VALIDATIONS = (
    "--benchmark-cold-index <folder> --output <json> [--performance]",
    "--compare-index <baseline-db> <candidate-db> --queries <json>",
)


def run_validations(executable: Path) -> int:
    executable = executable.resolve()
    if not executable.is_file():
        print(f"FROZEN_VALIDATION_FAILED: executable is missing: {executable}")
        return 1
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    validation_dir = executable.parent / "发行资料" / "验证结果"
    validation_dir.mkdir(parents=True, exist_ok=True)
    for argument in VALIDATIONS:
        print(f"[frozen-validation] {argument}", flush=True)
        result = subprocess.run(
            [str(executable), argument],
            cwd=validation_dir,
            env=environment,
            check=False,
        )
        if result.returncode != 0:
            print(
                "FROZEN_VALIDATION_FAILED "
                f"argument={argument} exit_code={result.returncode}"
            )
            return result.returncode or 1
    print(f"FROZEN_VALIDATION_OK commands={len(VALIDATIONS)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    args = parser.parse_args()
    return run_validations(args.executable)


if __name__ == "__main__":
    raise SystemExit(main())
