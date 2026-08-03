from __future__ import annotations

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.hardware_profile import (
    HardwareProfile,
    aggregate_disk_class,
    build_performance_profile,
    settings_for_profile,
)


def test_laptop_performance_profile_reserves_system_capacity() -> None:
    profile = build_performance_profile(
        HardwareProfile(
            logical_cpus=16,
            physical_cores=12,
            memory_total_mb=16_384,
            memory_available_mb=12_000,
                disk_class="ssd",
        )
    )

    assert profile.cpu_token_budget == 14
    assert 7_500 <= profile.memory_budget_mb <= 8_192
    assert profile.normal_workers == 7
    assert profile.office_workers == 3
    assert profile.pdf_workers == 2
    assert profile.zip_member_workers == 3
    assert profile.ocr_workers == 2


def test_constrained_disk_and_memory_reduce_profile() -> None:
    profile = build_performance_profile(
        HardwareProfile(
            logical_cpus=8,
            physical_cores=4,
            memory_total_mb=8192,
            memory_available_mb=1500,
            disk_class="network",
        )
    )

    assert profile.cpu_token_budget == 6
    assert profile.memory_budget_mb == 1024
    assert profile.normal_workers <= 3
    assert profile.office_workers == 1
    assert profile.pdf_workers == 1
    assert profile.zip_member_workers == 1
    assert profile.ocr_workers == 1


def test_profile_settings_are_a_copy_and_use_resource_limits() -> None:
    original = AppSettings(parser_workers=2, index_memory_budget_mb=2048)
    profile = build_performance_profile(
        HardwareProfile(16, 12, 16_384, 12_000, "local")
    )

    effective = settings_for_profile(original, profile)

    assert original.parser_workers == 2
    assert effective.parser_workers == profile.normal_workers
    assert effective.process_parser_workers == profile.office_workers
    assert effective.pdf_parser_workers == profile.pdf_workers
    assert effective.slow_file_workers == profile.zip_member_workers
    assert effective.index_memory_budget_mb == profile.memory_budget_mb
    assert effective.index_cpu_token_budget == profile.cpu_token_budget
    assert effective.index_performance_preset == "fastest"
    assert effective.pdf_page_batch_size >= 16
    assert effective.ocr_pending_tasks >= effective.ocr_microbatch_parent_jobs


def test_high_memory_performance_profile_is_not_capped_at_eight_gib() -> None:
    profile = build_performance_profile(
        HardwareProfile(32, 24, 65_536, 49_152, "ssd")
    )

    assert profile.memory_budget_mb > 8192
    assert profile.cpu_token_budget > 14
    assert profile.pdf_workers >= 3
    assert profile.ocr_workers >= 2


def test_unknown_local_disk_does_not_force_single_worker_profile() -> None:
    profile = build_performance_profile(
        HardwareProfile(16, 12, 16_384, 12_000, "unknown")
    )

    assert profile.pdf_workers >= 2
    assert profile.ocr_workers >= 2


def test_multiple_roots_use_the_most_constrained_disk_class() -> None:
    assert aggregate_disk_class(["ssd", "hdd"]) == "hdd"
    assert aggregate_disk_class(["ssd", "network"]) == "network"
    assert aggregate_disk_class(["ssd", "removable"]) == "removable"
    assert aggregate_disk_class(["ssd", "ssd"]) == "ssd"
