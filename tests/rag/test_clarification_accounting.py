from local_full_text_search.rag.evaluation.runner import _api_attempt_count
from local_full_text_search.rag.evaluation.reporting.timing import case_timing


def test_clarification_call_is_counted_without_a_generation():
    events = [{'stage': s} for s in ('assessing_clarification', 'clarification_assessed', 'needs_clarification')]
    assert _api_attempt_count(events) == 1
    assert _api_attempt_count([*events, {'stage': 'clarification_retry'}]) == 2


def test_clarification_time_is_separate_from_answer_generation():
    timing = case_timing({'duration_ms': 70, 'events': [
        {'stage': 'assessing_clarification', 'elapsed_ms': 0},
        {'stage': 'clarification_assessed', 'elapsed_ms': 20},
        {'stage': 'generating', 'elapsed_ms': 25},
        {'stage': 'completed', 'elapsed_ms': 70},
    ]})
    assert timing['modules_ms']['clarification'] == 20
    assert timing['modules_ms']['generation'] == 45
    assert sum(timing['modules_ms'].values()) == 70
