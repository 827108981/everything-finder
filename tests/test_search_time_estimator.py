from local_full_text_search.core.search_time_estimator import (
    SearchEstimateContext,
    SearchTimeEstimator,
)


def context(**changes: object) -> SearchEstimateContext:
    values = {
        "mode": "exact",
        "file_count": 20_000,
        "scoped": False,
        "extension_filtered": False,
        "searches_content": True,
        "ocr_fuzzy": False,
        "case_sensitive": False,
    }
    values.update(changes)
    return SearchEstimateContext(**values)


def test_regex_and_large_indexes_get_longer_estimates() -> None:
    estimator = SearchTimeEstimator()

    ordinary = estimator.estimate(context())
    expensive = estimator.estimate(context(mode="regex", file_count=200_000))

    assert expensive.lower_ms > ordinary.lower_ms
    assert expensive.upper_ms > ordinary.upper_ms


def test_scope_and_file_type_filters_reduce_estimate() -> None:
    estimator = SearchTimeEstimator()

    unfiltered = estimator.estimate(context())
    filtered = estimator.estimate(context(scoped=True, extension_filtered=True))

    assert filtered.upper_ms < unfiltered.upper_ms


def test_completed_searches_calibrate_the_same_workload() -> None:
    estimator = SearchTimeEstimator()
    workload = context()
    initial = estimator.estimate(workload)

    estimator.observe(workload, 4_000)
    calibrated = estimator.estimate(workload)

    assert calibrated.sample_count == 1
    assert calibrated.upper_ms > initial.upper_ms
    assert "预计" in calibrated.display_text()
