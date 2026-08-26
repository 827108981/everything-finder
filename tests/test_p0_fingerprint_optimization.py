from pathlib import Path

from local_full_text_search.core.index_manager import choose_duplicate_strategy


def test_p0_dynamic_strategy_does_not_hash_unique_source() -> None:
    assert not choose_duplicate_strategy(
        Path("notes.txt"),
        4 * 1024 * 1024,
        group_size=1,
        has_archive=False,
    )


def test_p0_dynamic_strategy_hashes_expensive_duplicate_group() -> None:
    assert choose_duplicate_strategy(
        Path("manual.pdf"),
        2 * 1024 * 1024,
        group_size=2,
        has_archive=False,
    )


def test_p0_dynamic_strategy_keeps_archive_candidate_exact() -> None:
    assert choose_duplicate_strategy(
        Path("member.txt"),
        128,
        group_size=1,
        has_archive=True,
    )
