"""在创建索引候选数据库前，以只读方式判断索引范围是否发生变化。"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.scanner import iter_files
from local_full_text_search.core.task_manager import CancelToken


@dataclass(frozen=True, slots=True)
class IndexChangeCheck:
    """候选库创建前的只读检查结果。"""

    unchanged: bool
    reason: str
    active_generation: int | None = None
    checked_roots: int = 0
    checked_files: int = 0
    changed_files: int = 0
    elapsed_ms: int = 0
    scan_signature: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def check_published_index_changes(
    store: IndexGenerationStore,
    control_db_path: Path,
    settings: AppSettings,
    *,
    root_ids: Sequence[int] = (),
    file_ids: Sequence[int] = (),
    parser_identity: Callable[[Path, AppSettings], tuple[str, str]],
    cancel_token: CancelToken | None = None,
) -> IndexChangeCheck:
    """只在能确认无变化时返回 unchanged=True；任何不确定情况都继续正常索引。

    比较与正式索引器相同的文件身份（size + mtime_ns）、扫描范围和解析器版本，
    不读取正文、不写 SQLite，也不会创建 *.building.db。完整扫描的文件清单签名
    写入 generation manifest；后续即使有无法解析的文件，也能稳定识别它并避免
    每次启动都复制一份数据库。
    """

    started = time.monotonic()
    token = cancel_token or CancelToken()

    def result(
        unchanged: bool,
        reason: str,
        *,
        generation: int | None = None,
        roots: int = 0,
        files: int = 0,
        changed_files: int = 0,
        signature: str | None = None,
    ) -> IndexChangeCheck:
        return IndexChangeCheck(
            unchanged,
            reason,
            generation,
            roots,
            files,
            changed_files,
            int((time.monotonic() - started) * 1000),
            signature,
        )

    if file_ids:
        return result(False, "explicit_file_retry")
    if store.build_lock_path.exists():
        return result(False, "another_candidate_is_building")
    active = store.active()
    if active is None:
        return result(False, "no_published_generation")
    if settings.retry_failed_files:
        # 该选项明确要求本轮重新尝试失败文件；索引快照不保留失败文件清单，
        # 因而不能安全地把这次请求优化成无操作。
        return result(False, "retry_failed_files_enabled", generation=active.generation)

    try:
        control = DatabaseManager(Path(control_db_path), read_only=True)
        baseline = DatabaseManager(active.path, read_only=True)
        control_roots = [dict(row) for row in control.list_roots()]
        baseline_roots = [dict(row) for row in baseline.list_roots()]
        selected_ids = {int(value) for value in root_ids}
        enabled_roots = [row for row in control_roots if bool(row.get("enabled"))]
        if selected_ids:
            enabled_roots = [row for row in enabled_roots if int(row["id"]) in selected_ids]

        expected_rows = _load_physical_file_rows(baseline)
        expected_by_path: dict[str, dict[str, object]] = {}
        selected_root_set = {int(row["id"]) for row in enabled_roots}
        for row in expected_rows:
            if int(row["root_id"]) in selected_root_set:
                expected_by_path[_normalized_path(str(row["path"]))] = row

        scope_changed = _root_signature(control_roots) != _root_signature(baseline_roots)
        inventory: list[tuple[object, ...]] = []
        found_paths: set[str] = set()
        checked_files = 0
        changed_files = 0
        for root in enabled_roots:
            token.throw_if_cancelled()
            root_path = Path(str(root["path"]))
            if not root_path.is_dir():
                return result(
                    False,
                    "configured_root_unavailable",
                    generation=active.generation,
                    roots=len(enabled_roots),
                    files=checked_files,
                    changed_files=changed_files + 1,
                )
            for file_path in iter_files(
                root_path,
                include_subfolders=bool(root.get("include_subfolders", True)),
                settings=settings,
                cancel_token=token,
            ):
                token.throw_if_cancelled()
                normalized = _normalized_path(str(file_path))
                found_paths.add(normalized)
                checked_files += 1
                try:
                    stat = file_path.stat()
                    parser_name, parser_version = parser_identity(file_path, settings)
                except OSError:
                    return result(
                        False,
                        "file_stat_failed",
                        generation=active.generation,
                        roots=len(enabled_roots),
                        files=checked_files,
                        changed_files=changed_files + 1,
                    )
                except Exception:
                    return result(
                        False,
                        "parser_identity_failed",
                        generation=active.generation,
                        roots=len(enabled_roots),
                        files=checked_files,
                        changed_files=changed_files + 1,
                    )
                root_id = int(root["id"])
                inventory.append(
                    (root_id, normalized, int(stat.st_size), int(stat.st_mtime_ns), parser_name, parser_version)
                )

                previous = expected_by_path.get(normalized)
                if previous is None:
                    changed_files += 1
                    continue
                if (
                    int(previous.get("size_bytes") or 0) != int(stat.st_size)
                    or str(previous.get("quick_fingerprint") or "")
                    != f"{int(stat.st_size)}:{int(stat.st_mtime_ns)}"
                ):
                    changed_files += 1
                    continue
                if bool(previous.get("manually_excluded")):
                    continue
                if (
                    str(previous.get("parser_name") or "") != parser_name
                    or str(previous.get("parser_version") or "") != parser_version
                ):
                    changed_files += 1
                    continue
                status = str(previous.get("parse_status") or "")
                if status in {"pending", "processing", "cancelled", "deleted"}:
                    changed_files += 1
                elif (
                    settings.retry_failed_files
                    and status != "success"
                    and not (
                        str(previous.get("extension") or "") in {".mp4", ".mov", ".avi", ".mkv"}
                        and status == "metadata_only"
                    )
                ):
                    changed_files += 1

        changed_files += len(set(expected_by_path) - found_paths)
        full_scan = not selected_ids
        signature = _make_signature(control_roots, settings, inventory) if full_scan else None

        # 完整清单签名是最高效且能覆盖已知失败文件的判断；必须与当前正式版本匹配。
        published_signature = store.scan_signature() if full_scan else None
        if full_scan and signature and published_signature == signature:
            latest = store.active()
            if latest is not None and latest.generation == active.generation and not store.build_lock_path.exists():
                return result(
                    True,
                    "published_inventory_signature_matches",
                    generation=active.generation,
                    roots=len(enabled_roots),
                    files=checked_files,
                    signature=signature,
                )
            return result(
                False,
                "active_generation_changed_during_check",
                generation=active.generation,
                roots=len(enabled_roots),
                files=checked_files,
                changed_files=max(changed_files, 1),
                signature=signature,
            )

        if scope_changed or changed_files:
            return result(
                False,
                "root_scope_changed" if scope_changed else "filesystem_or_parser_changed",
                generation=active.generation,
                roots=len(enabled_roots),
                files=checked_files,
                changed_files=max(changed_files, int(scope_changed)),
                signature=signature,
            )

        # 已有签名但不一致，说明即使文件大小/时间戳相同，索引策略也变了
        # （例如正文块长度或完整哈希设置）。不能只更新签名，否则会把旧内容
        # 错当成由新设置生成；应走正常候选库流程重新生成。
        if full_scan and signature and published_signature:
            return result(
                False,
                "scan_signature_changed",
                generation=active.generation,
                roots=len(enabled_roots),
                files=checked_files,
                changed_files=1,
                signature=signature,
            )

        # 遷移舊 manifest：確認資料庫與磁碟完全一致後只更新 manifest，不新建資料庫。
        if full_scan and signature:
            persisted = store.set_scan_signature(active.generation, signature)
            latest = store.active()
            if not persisted or latest is None or latest.generation != active.generation:
                return result(
                    False,
                    "active_generation_changed_during_check",
                    generation=active.generation,
                    roots=len(enabled_roots),
                    files=checked_files,
                    changed_files=1,
                    signature=signature,
                )
        return result(
            True,
            "no_filesystem_scope_or_parser_changes",
            generation=active.generation,
            roots=len(enabled_roots),
            files=checked_files,
            signature=signature,
        )
    except (OSError, sqlite3.Error, RuntimeError, ValueError, TypeError):
        # 预检失败不能误报“无变化”；交给原有事务化索引流程进一步处理并报告错误。
        return result(False, "preflight_uncertain", generation=active.generation)


def _root_signature(rows: list[dict[str, object]]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        sorted(
            (
                int(row.get("id") or 0),
                _normalized_path(str(row.get("path") or "")),
                int(row.get("enabled") or 0),
                int(row.get("include_subfolders") or 0),
                str(row.get("exclude_rules_json") or "{}"),
            )
            for row in rows
        )
    )


def _settings_signature(settings: AppSettings) -> dict[str, object]:
    return {
        "excluded_dirs": sorted(str(value) for value in (settings.excluded_dirs or ())),
        "excluded_file_patterns": sorted(
            str(value) for value in (settings.excluded_file_patterns or ())
        ),
        "include_hidden_files": bool(settings.include_hidden_files),
        "fast_ooxml_enabled": bool(settings.fast_ooxml_enabled),
        "block_target_chars": int(settings.block_target_chars),
        "block_max_chars": int(settings.block_max_chars),
        "compute_full_hash": bool(settings.compute_full_hash),
        "enable_ocr": bool(settings.enable_ocr),
        "ocr_language": str(settings.ocr_language),
        "ocr_images": bool(settings.ocr_images),
        "ocr_scanned_pdf": bool(settings.ocr_scanned_pdf),
        "ocr_min_confidence": float(settings.ocr_min_confidence),
        "min_ocr_image_pixels": int(settings.min_ocr_image_pixels),
        "max_ocr_image_side": int(settings.max_ocr_image_side),
        "max_zip_depth": int(settings.max_zip_depth),
        "max_zip_file_count": int(settings.max_zip_file_count),
        "max_zip_uncompressed_bytes": int(settings.max_zip_uncompressed_bytes),
    }


def _make_signature(
    roots: list[dict[str, object]],
    settings: AppSettings,
    inventory: list[tuple[object, ...]],
) -> str:
    payload = {
        "roots": _root_signature(roots),
        "settings": _settings_signature(settings),
        "files": sorted(inventory),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8", errors="surrogatepass")).hexdigest()


def _load_physical_file_rows(database: DatabaseManager) -> list[dict[str, object]]:
    with database.connect_read_only() as con:
        columns = {
            str(row[1]) for row in con.execute("PRAGMA table_info(files)").fetchall()
        }
        exclusion_table = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='index_scope_exclusions'"
        ).fetchone()
        excluded_expr = (
            "EXISTS (SELECT 1 FROM index_scope_exclusions e "
            "WHERE e.file_id = f.id AND e.revoked_at IS NULL AND e.invalidated_at IS NULL)"
            if exclusion_table is not None
            else "0"
        )
        predicates = ["f.container_file_id IS NULL"]
        if "is_deleted" in columns:
            predicates.append("f.is_deleted = 0")
        optional = {
            name: name if name in columns else f"'' AS {name}"
            for name in ("filename", "extension", "quick_fingerprint", "parse_status", "parser_name", "parser_version")
        }
        size_expr = "f.size_bytes" if "size_bytes" in columns else "0 AS size_bytes"
        query = f"""
            SELECT f.id, f.root_id, f.path, {optional['filename']}, {optional['extension']},
                   {size_expr}, {optional['quick_fingerprint']}, {optional['parse_status']},
                   {optional['parser_name']}, {optional['parser_version']},
                   {excluded_expr} AS manually_excluded
            FROM files f WHERE {' AND '.join(predicates)}
        """
        rows = con.execute(query).fetchall()
    return [dict(row) for row in rows]


def _normalized_path(value: str) -> str:
    return os.path.normcase(os.path.abspath(value)).rstrip("\\/")
