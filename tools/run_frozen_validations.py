from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path


VALIDATIONS = (
    "--self-test",
    "--validate-partial-search",
    "--validate-search-index-coordination",
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
    "--validate-database-migration",
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
    # Never let frozen smoke tests open or mutate the user's real database,
    # logs, single-instance mutex and startup diagnostics. Keep the directory
    # after validation so a failed command leaves evidence for investigation.
    isolated_data_dir = Path(
        tempfile.mkdtemp(prefix="运行数据-", dir=str(validation_dir))
    )
    environment["LFTS_APP_DATA_DIR"] = str(isolated_data_dir)
    print(f"[frozen-validation] isolated_data_dir={isolated_data_dir}", flush=True)
    for argument in VALIDATIONS:
        attempts = 3 if argument == "--validate-hang-recovery" else 1
        result = None
        for attempt in range(attempts):
            if attempt:
                # Process-pool children from the preceding cold start may
                # still be leaving the OS process table. Give Windows a short
                # grace period before repeating the 1-second watchdog gate.
                time.sleep(10)
            print(
                f"[frozen-validation] {argument}"
                + (" (retry)" if attempt else ""),
                flush=True,
            )
            result = subprocess.run(
                [str(executable), argument],
                cwd=validation_dir,
                env=environment,
                check=False,
            )
            if result.returncode == 0:
                break
        assert result is not None
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
