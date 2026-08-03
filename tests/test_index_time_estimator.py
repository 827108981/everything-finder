from local_full_text_search.core.index_time_estimator import IndexTimeEstimator


def test_slow_lane_controls_parallel_critical_path() -> None:
    estimator = IndexTimeEstimator(
        {"normal": 100, "ocr": 100},
        {"normal": 4, "ocr": 1},
    )

    estimate = estimator.estimate(
        {"normal": 100, "ocr": 100},
        {"normal": 0, "ocr": 0},
        now=0,
    )

    assert estimate is not None
    assert estimate.seconds >= 100
    assert not estimate.ready
    assert estimate.display_text() == "正在估算…"


def test_recent_median_and_ewma_make_a_single_stable_estimate() -> None:
    estimator = IndexTimeEstimator({"ocr": 300}, {"ocr": 1})
    estimator.observe("ocr", 10, 20)
    estimator.observe("ocr", 10, 22)
    estimator.observe("ocr", 10, 18)

    estimate = estimator.estimate({"ocr": 100}, {"ocr": 0}, now=0)

    assert estimate is not None
    assert estimate.ready
    assert estimate.seconds > 60
    assert "约" in estimate.display_text()
    assert "-" not in estimate.display_text()


def test_overrunning_active_task_corrects_an_optimistic_static_prior() -> None:
    estimator = IndexTimeEstimator({"office_process": 1}, {"office_process": 1})

    estimate = estimator.estimate(
        {"office_process": 1},
        {"office_process": 300},
        now=0,
    )

    assert estimate is not None
    assert estimate.seconds >= 60


def test_display_updates_at_most_every_ten_seconds_and_limits_normal_jump() -> None:
    estimator = IndexTimeEstimator({"ocr": 100}, {"ocr": 1})
    for _ in range(3):
        estimator.observe("ocr", 10, 10)
    first = estimator.estimate({"ocr": 100}, now=0)
    assert first is not None
    for _ in range(3):
        estimator.observe("ocr", 10, 100)
    before_refresh = estimator.estimate({"ocr": 100}, now=5)
    after_refresh = estimator.estimate({"ocr": 100}, now=11)

    assert before_refresh is not None and after_refresh is not None
    assert before_refresh.seconds == first.seconds
    assert after_refresh.seconds <= first.seconds + max(5, int(first.seconds * 0.20))


def test_paused_estimator_freezes_until_resume() -> None:
    estimator = IndexTimeEstimator({"ocr": 100}, {"ocr": 1})
    for _ in range(3):
        estimator.observe("ocr", 10, 10)
    before = estimator.estimate({"ocr": 100}, now=0)
    assert before is not None
    estimator.pause()
    frozen = estimator.estimate({"ocr": 1}, now=100)
    estimator.resume()
    resumed = estimator.estimate({"ocr": 1}, now=101, force_recalibration=True)

    assert frozen == before
    assert resumed is not None
    assert resumed.seconds != frozen.seconds
