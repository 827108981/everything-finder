from local_full_text_search.core.index_time_estimator import IndexTimeEstimator


def test_slow_lane_controls_parallel_critical_path() -> None:
    estimator = IndexTimeEstimator(
        {"normal": 100, "ocr": 100},
        {"normal": 4, "ocr": 1},
    )

    estimate = estimator.estimate(
        {"normal": 100, "ocr": 100},
        {"normal": 0, "ocr": 0},
    )

    assert estimate is not None
    assert estimate.upper_seconds >= 180


def test_early_fast_samples_do_not_collapse_ocr_estimate() -> None:
    estimator = IndexTimeEstimator({"ocr": 300}, {"ocr": 1})
    estimator.observe("ocr", 100, 1)

    estimate = estimator.estimate({"ocr": 200}, {"ocr": 0})

    assert estimate is not None
    assert estimate.lower_seconds >= 120
    assert estimate.upper_seconds >= 360


def test_overrunning_active_task_expands_tiny_static_estimate() -> None:
    estimator = IndexTimeEstimator({"office_process": 1}, {"office_process": 1})

    estimate = estimator.estimate(
        {"office_process": 1},
        {"office_process": 300},
    )

    assert estimate is not None
    assert estimate.lower_seconds >= 30
    assert estimate.upper_seconds >= 450
