from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path

from local_full_text_search.core.ocr_backend_benchmark import (
    BackendTrial,
    candidate_backends,
    choose_default_backend,
)


def _trial(
    backend: str,
    elapsed_ms: int,
    *,
    digest: str = "same",
    rss: int = 1_000,
    offline: bool = True,
) -> BackendTrial:
    return BackendTrial(
        backend=backend,
        elapsed_ms=elapsed_ms,
        model_load_ms=100,
        first_batch_ms=20,
        peak_rss_bytes=rss,
        accuracy_digest=digest,
        text_digest=digest,
        box_digest=digest,
        confidence_digest=digest,
        offline=offline,
        error="",
    )


def test_p1_01r_declares_all_required_backend_candidates() -> None:
    candidates = candidate_backends()

    assert [candidate.backend for candidate in candidates] == [
        "paddle_cpu_mkldnn_off",
        "paddle_cpu_mkldnn_on",
        "onnxruntime_cpu",
        "openvino_cpu",
    ]
    assert all(candidate.same_model_required for candidate in candidates)


def test_p1_01r_backend_switch_requires_three_runs_accuracy_and_20_percent() -> None:
    baseline = [
        _trial("paddle_cpu_mkldnn_off", value)
        for value in (1000, 1050, 950)
    ]
    fast_accurate = [
        _trial("paddle_cpu_mkldnn_on", value)
        for value in (760, 780, 770)
    ]
    inaccurate = [
        _trial("onnxruntime_cpu", value, digest="different")
        for value in (500, 510, 490)
    ]
    too_few = [_trial("openvino_cpu", 400)]

    decision = choose_default_backend(
        baseline,
        [fast_accurate, inaccurate, too_few],
        rss_budget_bytes=4 * 1024**3,
    )

    assert decision.selected_backend == "paddle_cpu_mkldnn_on"
    assert decision.speedup_ratio >= 0.20
    assert decision.rejections["onnxruntime_cpu"] == "accuracy_digest_mismatch"
    assert decision.rejections["openvino_cpu"] == "requires_three_runs"


def test_p1_01r_keeps_baseline_when_no_candidate_passes_all_gates() -> None:
    baseline = [
        _trial("paddle_cpu_mkldnn_off", value)
        for value in (1000, 1000, 1000)
    ]
    only_ten_percent = [
        _trial("paddle_cpu_mkldnn_on", value)
        for value in (900, 900, 900)
    ]

    decision = choose_default_backend(
        baseline,
        [only_ten_percent],
        rss_budget_bytes=4 * 1024**3,
    )

    assert decision.selected_backend == "paddle_cpu_mkldnn_off"
    assert decision.changed is False
    assert (
        decision.rejections["paddle_cpu_mkldnn_on"]
        == "speedup_below_20_percent"
    )


def test_p1_01r_never_selects_a_failed_baseline_trial() -> None:
    baseline = [
        replace(
            _trial("paddle_cpu_mkldnn_off", 0),
            error="TrialTimeout",
        )
        for _ in range(3)
    ]

    decision = choose_default_backend(
        baseline,
        [],
        rss_budget_bytes=4 * 1024**3,
    )

    assert decision.selected_backend == ""
    assert (
        decision.rejections["paddle_cpu_mkldnn_off"]
        == "baseline_backend_error"
    )


def test_p1_01r_allows_harmless_confidence_float_differences() -> None:
    baseline = [
        replace(
            _trial("paddle_cpu_mkldnn_off", value),
            confidence_digest="baseline-confidence",
            confidence_mean=0.91,
            confidence_min=0.84,
        )
        for value in (1000, 1020, 980)
    ]
    candidate = [
        replace(
            _trial("paddle_cpu_mkldnn_on", value),
            confidence_digest="candidate-confidence",
            confidence_mean=0.909,
            confidence_min=0.839,
        )
        for value in (700, 710, 690)
    ]

    decision = choose_default_backend(
        baseline,
        [candidate],
        rss_budget_bytes=4 * 1024**3,
    )

    assert decision.selected_backend == "paddle_cpu_mkldnn_on"
    assert decision.changed is True


def test_p1_01r_isolated_trial_drains_large_pipe_before_waiting_for_exit() -> None:
    tool_path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "benchmark_ocr_backends.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_ocr_backends_tool",
        tool_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events: list[str] = []

    class FakeReceiver:
        def poll(self, timeout: float = 0.0) -> bool:
            events.append("poll")
            return True

        def recv(self) -> tuple[str, dict[str, str]]:
            events.append("recv")
            return "ok", {"text": "x" * 1_000_000}

    class FakeProcess:
        exitcode = 0

        def join(self, timeout: float | None = None) -> None:
            events.append("join")

        def is_alive(self) -> bool:
            return False

    message, timed_out = module._receive_trial_message(
        FakeProcess(),
        FakeReceiver(),
        timeout_seconds=1.0,
    )

    assert timed_out is False
    assert message[0] == "ok"
    assert events.index("recv") < events.index("join")


def test_p1_01r_background_trial_detects_an_invalid_console_stream() -> None:
    tool_path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "benchmark_ocr_backends.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_ocr_backends_tool_console",
        tool_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class InvalidConsole:
        def fileno(self) -> int:
            raise OSError(22, "Invalid argument")

    assert module._standard_stream_is_usable(InvalidConsole()) is False
