from __future__ import annotations

import ctypes
import os
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    logical_cpus: int
    physical_cores: int | None
    memory_total_mb: int
    memory_available_mb: int
    disk_class: str


@dataclass(frozen=True, slots=True)
class RunResourceProfile:
    mode: str
    logical_cpus: int
    physical_cores: int | None
    memory_budget_mb: int
    cpu_token_budget: int
    normal_workers: int
    office_workers: int
    zip_member_workers: int
    ocr_workers: int
    ocr_cpu_threads: int
    legacy_office_workers: int
    db_write_batch_blocks: int
    db_write_batch_bytes: int
    disk_class: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_hardware_profile(index_path: Path | None = None) -> HardwareProfile:
    logical_cpus = max(1, int(os.cpu_count() or 1))
    physical_cores: int | None = None
    memory_total_mb = 0
    memory_available_mb = 0
    try:
        import psutil

        physical_cores = psutil.cpu_count(logical=False)
        memory = psutil.virtual_memory()
        memory_total_mb = max(0, int(memory.total // (1024 * 1024)))
        memory_available_mb = max(0, int(memory.available // (1024 * 1024)))
    except Exception:
        pass
    return HardwareProfile(
        logical_cpus=logical_cpus,
        physical_cores=physical_cores,
        memory_total_mb=memory_total_mb,
        memory_available_mb=memory_available_mb,
        disk_class=classify_disk(index_path),
    )


def classify_disk(path: Path | None) -> str:
    if path is None:
        return "unknown"
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if os.name != "nt":
        return "unknown"
    drive = resolved.drive
    if not drive:
        return "unknown"
    try:
        drive_type = ctypes.windll.kernel32.GetDriveTypeW(f"{drive}\\")
    except Exception:
        return "unknown"
    drive_class = {
        2: "removable",
        3: _local_disk_media_type(drive),
        4: "network",
    }.get(int(drive_type), "unknown")
    return drive_class


def _local_disk_media_type(drive: str) -> str:
    """Resolve a local Windows volume to SSD/HDD when PowerShell can provide it.

    This runs only after the user explicitly starts performance mode, so a
    short WMI/Storage query cannot slow ordinary application startup.
    """

    drive_letter = drive.rstrip("\\/").rstrip(":")
    if not drive_letter:
        return "unknown"
    command = (
        "$ErrorActionPreference='Stop'; "
        f"(Get-Partition -DriveLetter '{drive_letter}' | Get-Disk | "
        "Select-Object -First 1 -ExpandProperty MediaType)"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    media_type = result.stdout.strip().upper()
    if "SSD" in media_type:
        return "ssd"
    if "HDD" in media_type:
        return "hdd"
    return "unknown"


def build_performance_profile(hardware: HardwareProfile) -> RunResourceProfile:
    logical = max(1, hardware.logical_cpus)
    physical = max(1, hardware.physical_cores or logical)
    constrained_disk = hardware.disk_class in {"hdd", "network", "removable", "unknown"}
    cpu_cap = max(2, logical - 2)
    cpu_cap = min(cpu_cap, 14 if not constrained_disk else 6)
    total_memory = hardware.memory_total_mb or 4096
    available_memory = hardware.memory_available_mb or total_memory
    memory_budget = min(8192, total_memory // 2, max(1024, available_memory - 2048))
    if available_memory < 2048:
        memory_budget = min(memory_budget, 1024)
    elif available_memory < 4096:
        memory_budget = min(memory_budget, 2048)
    if constrained_disk:
        memory_budget = min(memory_budget, 2048)
    normal_workers = min(6, max(2, cpu_cap // 2))
    office_workers = min(3, max(1, physical // 4))
    zip_workers = min(3, max(1, physical // 4))
    ocr_workers = 2 if memory_budget >= 4096 and cpu_cap >= 8 else 1
    ocr_threads = 2 if cpu_cap >= 6 else 1
    if constrained_disk:
        normal_workers = min(normal_workers, 3)
        office_workers = 1
        zip_workers = 1
        ocr_workers = 1
    return RunResourceProfile(
        mode="performance",
        logical_cpus=logical,
        physical_cores=hardware.physical_cores,
        memory_budget_mb=memory_budget,
        cpu_token_budget=cpu_cap,
        normal_workers=normal_workers,
        office_workers=office_workers,
        zip_member_workers=zip_workers,
        ocr_workers=ocr_workers,
        ocr_cpu_threads=ocr_threads,
        legacy_office_workers=1,
        db_write_batch_blocks=4000 if memory_budget >= 4096 else 2000,
        db_write_batch_bytes=32 * 1024 * 1024 if memory_budget >= 4096 else 16 * 1024 * 1024,
        disk_class=hardware.disk_class,
    )


def settings_for_profile(settings: AppSettings, profile: RunResourceProfile) -> AppSettings:
    process_memory_budget = max(512, min(2048, profile.memory_budget_mb // max(2, profile.office_workers)))
    return replace(
        settings,
        parser_workers=profile.normal_workers,
        ocr_workers=profile.ocr_workers,
        ocr_cpu_threads=profile.ocr_cpu_threads,
        slow_file_workers=profile.zip_member_workers,
        process_parser_workers=profile.office_workers,
        normal_pending_tasks=profile.normal_workers * 2,
        ocr_pending_tasks=profile.ocr_workers * 2,
        slow_pending_tasks=profile.zip_member_workers * 2,
        process_pending_tasks=profile.office_workers * 2,
        max_pending_parse_tasks=max(16, profile.cpu_token_budget * 4),
        index_memory_budget_mb=profile.memory_budget_mb,
        index_cpu_token_budget=profile.cpu_token_budget,
        process_memory_budget_mb=process_memory_budget,
        normal_inflight_bytes=max(128 * 1024 * 1024, profile.memory_budget_mb * 1024 * 1024 // 4),
        office_inflight_bytes=max(256 * 1024 * 1024, profile.memory_budget_mb * 1024 * 1024 // 2),
        ocr_inflight_bytes=max(128 * 1024 * 1024, profile.memory_budget_mb * 1024 * 1024 // 4),
        slow_inflight_bytes=max(128 * 1024 * 1024, profile.memory_budget_mb * 1024 * 1024 // 4),
        index_write_batch_size=max(settings.index_write_batch_size, 64),
        db_write_batch_blocks=profile.db_write_batch_blocks,
        db_write_batch_bytes=profile.db_write_batch_bytes,
        index_performance_preset="balanced",
    )
