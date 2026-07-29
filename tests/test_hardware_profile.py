from __future__ import annotations

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.hardware_profile import (
    HardwareProfile,
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
    assert profile.memory_budget_mb == 8192
    assert profile.normal_workers == 6
    assert profile.office_workers == 3
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
    assert effective.slow_file_workers == profile.zip_member_workers
    assert effective.index_memory_budget_mb == profile.memory_budget_mb
    assert effective.index_cpu_token_budget == profile.cpu_token_budget
