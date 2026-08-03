from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import multiprocessing
import os
import statistics
import tempfile
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_full_text_search.core.ocr_backend_benchmark import (
    BackendTrial,
    candidate_backends,
    choose_default_backend,
)
from local_full_text_search.ocr.ocr_engine import OcrEngine


class _PeakRssSampler:
    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_PeakRssSampler":
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1)

    def _run(self) -> None:
        import psutil

        process = psutil.Process()
        while not self._stop.wait(0.05):
            try:
                rss = int(process.memory_info().rss) + sum(
                    int(child.memory_info().rss)
                    for child in process.children(recursive=True)
                    if child.is_running()
                )
                self.peak = max(self.peak, rss)
            except psutil.Error:
                continue


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _run_paddle(
    backend: str,
    inputs: list[Path],
    *,
    threads: int,
) -> BackendTrial:
    engine = OcrEngine(
        cpu_threads=threads,
        enable_mkldnn=backend.endswith("_on"),
    )
    outputs: list[dict[str, object]] = []
    first_batch_ms = 0
    started = time.perf_counter()
    with _PeakRssSampler() as sampler:
        for index, path in enumerate(inputs):
            item_started = time.perf_counter()
            result = engine.recognize_adaptive(path)
            if index == 0:
                first_batch_ms = int(
                    (time.perf_counter() - item_started) * 1000
                )
            outputs.append(
                {
                    "path": path.name,
                    "text": result.text,
                    "confidence": result.confidence,
                    "boxes": result.extra.get("boxes", []),
                }
            )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    text_payload = [item["text"] for item in outputs]
    box_payload = [item["boxes"] for item in outputs]
    confidence_payload = [item["confidence"] for item in outputs]
    return BackendTrial(
        backend=backend,
        elapsed_ms=elapsed_ms,
        model_load_ms=int(engine.model_load_ms),
        first_batch_ms=first_batch_ms,
        peak_rss_bytes=int(sampler.peak),
        accuracy_digest=_digest(outputs),
        text_digest=_digest(text_payload),
        box_digest=_digest(box_payload),
        confidence_digest=_digest(confidence_payload),
        offline=True,
        error="",
        confidence_mean=(
            float(statistics.mean(confidence_payload))
            if confidence_payload
            else 0.0
        ),
        confidence_min=(
            float(min(confidence_payload))
            if confidence_payload
            else 0.0
        ),
        text_samples=tuple(
            str(value) for value in text_payload
        ),
        confidence_samples=tuple(
            float(value) for value in confidence_payload
        ),
        box_digests_by_input=tuple(
            _digest(value) for value in box_payload
        ),
    )


def _trial_process_entry(
    sender: object,
    backend: str,
    input_paths: list[str],
    threads: int,
) -> None:
    _ensure_child_standard_streams()
    try:
        trial = _run_paddle(
            backend,
            [Path(path) for path in input_paths],
            threads=threads,
        )
        sender.send(("ok", asdict(trial)))
    except BaseException as exc:
        sender.send(
            (
                "error",
                (
                    f"{type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                ),
            )
        )
    finally:
        sender.close()


def _standard_stream_is_usable(stream: object) -> bool:
    if stream is None:
        return False
    try:
        descriptor = int(stream.fileno())
        os.fstat(descriptor)
        stream.write("")
        stream.flush()
    except (AttributeError, OSError, TypeError, ValueError):
        return False
    return True


def _ensure_child_standard_streams() -> None:
    for attribute in ("stdout", "stderr"):
        stream = getattr(sys, attribute, None)
        if _standard_stream_is_usable(stream):
            continue
        setattr(
            sys,
            attribute,
            open(
                os.devnull,
                "w",
                encoding="utf-8",
                errors="replace",
            ),
        )


def _run_isolated_trial(
    backend: str,
    inputs: list[Path],
    *,
    threads: int,
    timeout_seconds: float,
) -> BackendTrial:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_trial_process_entry,
        args=(
            sender,
            backend,
            [str(path) for path in inputs],
            int(threads),
        ),
        name=f"lfts-ocr-ab-{backend}",
    )
    process.start()
    sender.close()
    message, timed_out = _receive_trial_message(
        process,
        receiver,
        timeout_seconds=max(1.0, float(timeout_seconds)),
    )
    if timed_out:
        process.terminate()
        process.join(timeout=2)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=1)
        receiver.close()
        return _failed_trial(
            backend,
            (
                "TrialTimeout: single backend run exceeded "
                f"{float(timeout_seconds):g} seconds"
            ),
        )
    # Some native inference runtimes keep non-daemon cleanup threads alive
    # after the result has been serialized.  Once the complete result has
    # been drained from the pipe, terminate only that isolated trial process
    # rather than turning a valid benchmark into an apparent timeout.
    if message is not None and process.is_alive():
        process.terminate()
        process.join(timeout=2)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=1)
    try:
        if message is None:
            return _failed_trial(
                backend,
                (
                    "TrialProcessError: worker exited without a result "
                    f"(exit_code={process.exitcode})"
                ),
            )
    finally:
        receiver.close()
    if (
        isinstance(message, tuple)
        and len(message) == 2
        and message[0] == "ok"
        and isinstance(message[1], dict)
    ):
        return BackendTrial(**message[1])
    error = (
        str(message[1])
        if isinstance(message, tuple) and len(message) > 1
        else "TrialProcessError: invalid child result"
    )
    return _failed_trial(backend, error)


def _receive_trial_message(
    process: object,
    receiver: object,
    *,
    timeout_seconds: float,
) -> tuple[object | None, bool]:
    """Drain the result pipe before joining to avoid a large-payload deadlock."""

    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, True
        if receiver.poll(min(0.1, remaining)):
            message = receiver.recv()
            process.join(timeout=5)
            return message, False
        if not process.is_alive():
            process.join(timeout=0)
            if receiver.poll(0):
                return receiver.recv(), False
            return None, False


def _failed_trial(backend: str, error: str) -> BackendTrial:
    return BackendTrial(
        backend=backend,
        elapsed_ms=0,
        model_load_ms=0,
        first_batch_ms=0,
        peak_rss_bytes=0,
        accuracy_digest="",
        text_digest="",
        box_digest="",
        confidence_digest="",
        offline=True,
        error=str(error),
    )


def _materialize_inputs(
    paths: list[Path],
    output_dir: Path,
) -> list[Path]:
    images: list[Path] = []
    image_suffixes = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }
    for source in paths:
        if source.suffix.lower() in image_suffixes:
            images.append(source)
            continue
        if source.suffix.lower() != ".pdf":
            raise ValueError(f"Unsupported OCR benchmark input: {source}")
        import fitz

        document = fitz.open(source)
        try:
            for page_index in range(document.page_count):
                target = output_dir / (
                    f"{hashlib.sha256(str(source).encode()).hexdigest()[:12]}"
                    f"-page-{page_index + 1}.png"
                )
                document.load_page(page_index).get_pixmap(
                    dpi=200,
                    alpha=False,
                ).save(str(target))
                images.append(target)
        finally:
            document.close()
    return images


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run repeatable offline OCR backend A/B trials.",
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--rss-budget-gib", type=float, default=4.0)
    parser.add_argument(
        "--trial-timeout-seconds",
        type=float,
        default=900.0,
    )
    args = parser.parse_args()
    runs = max(3, int(args.runs))
    paths = [path.resolve() for path in args.inputs]
    for path in paths:
        if not path.is_file():
            parser.error(f"Input does not exist: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "schema_version": 2,
        "inputs": [str(path) for path in paths],
        "runs": runs,
        "threads": max(1, int(args.threads)),
        "trial_timeout_seconds": max(
            1.0,
            float(args.trial_timeout_seconds),
        ),
        "candidates": [
            asdict(candidate) for candidate in candidate_backends()
        ],
        "trials": {},
        "medians": {},
    }
    existing = _load_resumable_report(
        args.output,
        inputs=paths,
        runs=runs,
        threads=max(1, int(args.threads)),
    )
    if existing is not None:
        report["trials"] = dict(existing.get("trials") or {})
    with tempfile.TemporaryDirectory(prefix="lfts-ocr-backend-") as tmp:
        materialized = _materialize_inputs(paths, Path(tmp))
        trials_by_backend: dict[str, list[BackendTrial]] = {}
        for backend in (
            "paddle_cpu_mkldnn_off",
            "paddle_cpu_mkldnn_on",
        ):
            backend_trials = [
                BackendTrial(**item)
                for item in list(
                    dict(report["trials"]).get(backend) or []
                )[:runs]
                if isinstance(item, dict)
            ]
            while len(backend_trials) < runs:
                backend_trials.append(
                    _run_isolated_trial(
                        backend,
                        materialized,
                        threads=max(1, int(args.threads)),
                        timeout_seconds=max(
                            1.0,
                            float(args.trial_timeout_seconds),
                        ),
                    )
                )
                trials_by_backend[backend] = backend_trials
                _update_report(
                    report,
                    trials_by_backend,
                    runs=runs,
                    rss_budget_bytes=int(
                        max(
                            0.25,
                            float(args.rss_budget_gib),
                        )
                        * 1024**3
                    ),
                )
                _write_report_atomic(args.output, report)
            trials_by_backend[backend] = backend_trials
        _update_report(
            report,
            trials_by_backend,
            runs=runs,
            rss_budget_bytes=int(
                max(0.25, float(args.rss_budget_gib)) * 1024**3
            ),
        )
        report["non_paddle_evaluation"] = {
            "onnxruntime_cpu": (
                "未提供与包内 Paddle 权重逐指纹对应的离线 ONNX 模型，"
                "不能在同模型门禁下形成有效 A/B。"
            ),
            "openvino_cpu": (
                "未提供与包内 Paddle 权重逐指纹对应的离线 OpenVINO 模型，"
                "不能在同模型门禁下形成有效 A/B。"
            ),
        }
    _write_report_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    decision = dict(report["decision"])
    return 0 if decision.get("selected_backend") else 1


def _update_report(
    report: dict[str, object],
    trials_by_backend: dict[str, list[BackendTrial]],
    *,
    runs: int,
    rss_budget_bytes: int,
) -> None:
    report["trials"] = {
        backend: [asdict(trial) for trial in trials]
        for backend, trials in trials_by_backend.items()
    }
    report["medians"] = {
        backend: {
            "elapsed_ms": statistics.median(
                trial.elapsed_ms for trial in trials
            ),
            "peak_rss_bytes": max(
                (trial.peak_rss_bytes for trial in trials),
                default=0,
            ),
            "completed_runs": len(trials),
        }
        for backend, trials in trials_by_backend.items()
    }
    baseline = trials_by_backend.get(
        "paddle_cpu_mkldnn_off",
        [],
    )
    candidate = trials_by_backend.get(
        "paddle_cpu_mkldnn_on",
        [],
    )
    if len(baseline) >= runs and len(candidate) >= runs:
        report["decision"] = asdict(
            choose_default_backend(
                baseline,
                [candidate],
                rss_budget_bytes=rss_budget_bytes,
            )
        )
        report["accuracy_differences"] = (
            _accuracy_difference_report(
                baseline,
                candidate,
            )
        )
    else:
        report["decision"] = {
            "selected_backend": "",
            "changed": False,
            "speedup_ratio": 0.0,
            "rejections": {
                "benchmark": "pending_three_runs_per_backend"
            },
        }


def _load_resumable_report(
    output: Path,
    *,
    inputs: list[Path],
    runs: int,
    threads: int,
) -> dict[str, object] | None:
    if not output.is_file():
        return None
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        int(payload.get("schema_version") or 0) != 2
        or
        list(payload.get("inputs") or [])
        != [str(path) for path in inputs]
        or int(payload.get("runs") or 0) != int(runs)
        or int(payload.get("threads") or 0) != int(threads)
    ):
        return None
    return dict(payload)


def _accuracy_difference_report(
    baseline: list[BackendTrial],
    candidate: list[BackendTrial],
) -> dict[str, object]:
    baseline_first = baseline[0]
    candidate_first = candidate[0]
    per_input: list[dict[str, object]] = []
    count = max(
        len(baseline_first.text_samples),
        len(candidate_first.text_samples),
    )
    for index in range(count):
        baseline_text = (
            baseline_first.text_samples[index]
            if index < len(baseline_first.text_samples)
            else ""
        )
        candidate_text = (
            candidate_first.text_samples[index]
            if index < len(candidate_first.text_samples)
            else ""
        )
        difference = list(
            difflib.unified_diff(
                baseline_text.splitlines(),
                candidate_text.splitlines(),
                fromfile="baseline",
                tofile="candidate",
                lineterm="",
            )
        )[:60]
        baseline_confidence = (
            baseline_first.confidence_samples[index]
            if index
            < len(baseline_first.confidence_samples)
            else 0.0
        )
        candidate_confidence = (
            candidate_first.confidence_samples[index]
            if index
            < len(candidate_first.confidence_samples)
            else 0.0
        )
        baseline_box = (
            baseline_first.box_digests_by_input[index]
            if index
            < len(baseline_first.box_digests_by_input)
            else ""
        )
        candidate_box = (
            candidate_first.box_digests_by_input[index]
            if index
            < len(candidate_first.box_digests_by_input)
            else ""
        )
        per_input.append(
            {
                "input_index": index,
                "text_equal": baseline_text == candidate_text,
                "text_diff": difference,
                "box_equal": baseline_box == candidate_box,
                "baseline_confidence": baseline_confidence,
                "candidate_confidence": candidate_confidence,
                "confidence_delta": round(
                    candidate_confidence - baseline_confidence,
                    9,
                ),
            }
        )
    return {
        "text_digest_equal": (
            baseline_first.text_digest
            == candidate_first.text_digest
        ),
        "box_digest_equal": (
            baseline_first.box_digest
            == candidate_first.box_digest
        ),
        "confidence_digest_equal": (
            baseline_first.confidence_digest
            == candidate_first.confidence_digest
        ),
        "baseline_confidence_mean_median": statistics.median(
            trial.confidence_mean for trial in baseline
        ),
        "candidate_confidence_mean_median": statistics.median(
            trial.confidence_mean for trial in candidate
        ),
        "per_input": per_input,
    }


def _write_report_atomic(
    output: Path,
    report: dict[str, object],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(
        output.suffix + f".tmp.{os.getpid()}"
    )
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)


if __name__ == "__main__":
    raise SystemExit(main())
