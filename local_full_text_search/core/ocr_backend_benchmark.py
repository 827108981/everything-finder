from __future__ import annotations

import importlib.util
import statistics
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BackendCandidate:
    backend: str
    package: str
    available: bool
    same_model_required: bool
    technical_note: str


@dataclass(frozen=True, slots=True)
class BackendTrial:
    backend: str
    elapsed_ms: int
    model_load_ms: int
    first_batch_ms: int
    peak_rss_bytes: int
    accuracy_digest: str
    text_digest: str
    box_digest: str
    confidence_digest: str
    offline: bool
    error: str
    confidence_mean: float = 0.0
    confidence_min: float = 0.0
    text_samples: tuple[str, ...] = ()
    confidence_samples: tuple[float, ...] = ()
    box_digests_by_input: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BackendDecision:
    selected_backend: str
    changed: bool
    speedup_ratio: float
    rejections: dict[str, str]


def candidate_backends() -> list[BackendCandidate]:
    return [
        BackendCandidate(
            "paddle_cpu_mkldnn_off",
            "paddleocr",
            importlib.util.find_spec("paddleocr") is not None,
            True,
            "Current offline PP-OCRv4 Paddle inference path.",
        ),
        BackendCandidate(
            "paddle_cpu_mkldnn_on",
            "paddleocr",
            importlib.util.find_spec("paddleocr") is not None,
            True,
            "Same Paddle weights and preprocessing with MKL-DNN enabled.",
        ),
        BackendCandidate(
            "onnxruntime_cpu",
            "onnxruntime",
            importlib.util.find_spec("onnxruntime") is not None,
            True,
            (
                "Requires an offline, fingerprinted ONNX export of exactly the "
                "same detector and recognizer weights; package presence alone "
                "does not make the candidate eligible."
            ),
        ),
        BackendCandidate(
            "openvino_cpu",
            "openvino",
            importlib.util.find_spec("openvino") is not None,
            True,
            (
                "Requires an offline, fingerprinted OpenVINO conversion of "
                "exactly the same weights and equivalent preprocessing."
            ),
        ),
    ]


def choose_default_backend(
    baseline: list[BackendTrial],
    candidate_runs: list[list[BackendTrial]],
    *,
    rss_budget_bytes: int,
) -> BackendDecision:
    if len(baseline) < 3:
        raise ValueError("Baseline backend requires at least three runs")
    baseline_name = baseline[0].backend
    if any(trial.error for trial in baseline):
        return BackendDecision(
            selected_backend="",
            changed=False,
            speedup_ratio=0.0,
            rejections={
                baseline_name: "baseline_backend_error"
            },
        )
    if not all(trial.offline for trial in baseline):
        return BackendDecision(
            selected_backend="",
            changed=False,
            speedup_ratio=0.0,
            rejections={
                baseline_name: "baseline_offline_gate_failed"
            },
        )
    if any(trial.elapsed_ms <= 0 for trial in baseline):
        return BackendDecision(
            selected_backend="",
            changed=False,
            speedup_ratio=0.0,
            rejections={
                baseline_name: "baseline_invalid_elapsed"
            },
        )
    if any(
        trial.text_digest != baseline[0].text_digest
        or trial.box_digest != baseline[0].box_digest
        for trial in baseline
    ):
        return BackendDecision(
            selected_backend="",
            changed=False,
            speedup_ratio=0.0,
            rejections={
                baseline_name: "baseline_accuracy_unstable"
            },
        )
    if max(
        trial.peak_rss_bytes for trial in baseline
    ) > int(rss_budget_bytes):
        return BackendDecision(
            selected_backend="",
            changed=False,
            speedup_ratio=0.0,
            rejections={
                baseline_name: "baseline_rss_budget_exceeded"
            },
        )
    baseline_median = float(
        statistics.median(trial.elapsed_ms for trial in baseline)
    )
    selected = baseline_name
    selected_speedup = 0.0
    rejections: dict[str, str] = {}
    for trials in candidate_runs:
        if not trials:
            continue
        name = trials[0].backend
        if len(trials) < 3:
            rejections[name] = "requires_three_runs"
            continue
        if any(trial.error for trial in trials):
            rejections[name] = "backend_error"
            continue
        if not all(trial.offline for trial in trials):
            rejections[name] = "offline_gate_failed"
            continue
        if any(
            trial.text_digest != baseline[0].text_digest
            or trial.box_digest != baseline[0].box_digest
            for trial in trials
        ):
            rejections[name] = "accuracy_digest_mismatch"
            continue
        baseline_confidence = float(
            statistics.median(
                trial.confidence_mean for trial in baseline
            )
        )
        candidate_confidence = float(
            statistics.median(
                trial.confidence_mean for trial in trials
            )
        )
        baseline_min_confidence = float(
            statistics.median(
                trial.confidence_min for trial in baseline
            )
        )
        candidate_min_confidence = float(
            statistics.median(
                trial.confidence_min for trial in trials
            )
        )
        if (
            candidate_confidence + 0.02 < baseline_confidence
            or candidate_min_confidence + 0.02
            < baseline_min_confidence
        ):
            rejections[name] = "confidence_regression"
            continue
        if max(trial.peak_rss_bytes for trial in trials) > int(
            rss_budget_bytes
        ):
            rejections[name] = "rss_budget_exceeded"
            continue
        candidate_median = float(
            statistics.median(trial.elapsed_ms for trial in trials)
        )
        speedup = (
            (baseline_median - candidate_median) / baseline_median
            if baseline_median > 0
            else 0.0
        )
        if speedup < 0.20:
            rejections[name] = "speedup_below_20_percent"
            continue
        if speedup > selected_speedup:
            selected = name
            selected_speedup = speedup
    return BackendDecision(
        selected_backend=selected,
        changed=selected != baseline_name,
        speedup_ratio=round(selected_speedup, 6),
        rejections=rejections,
    )
