from __future__ import annotations

from local_full_text_search.core.foreground_activity import (
    acquire_search_lease,
    begin_search,
    end_search,
    search_active,
    wait_for_search_idle,
)


def test_search_activity_blocks_writer_only_while_search_is_active() -> None:
    end_search()
    assert wait_for_search_idle(0.01)
    begin_search()
    try:
        assert search_active()
        assert not wait_for_search_idle(0.01)
    finally:
        end_search()
    assert not search_active()
    assert wait_for_search_idle(0.05)


def test_search_lease_is_exclusive_and_releases_on_completion() -> None:
    end_search()
    first = acquire_search_lease("first", max_wait_seconds=0.05)
    try:
        assert search_active()
        try:
            acquire_search_lease("second", max_wait_seconds=0.01)
        except TimeoutError:
            pass
        else:
            raise AssertionError("a second foreground search acquired the lease")
    finally:
        first.release()
    assert not search_active()
    assert wait_for_search_idle(0.05)
