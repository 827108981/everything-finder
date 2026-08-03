from __future__ import annotations

import mmap
import re
import tempfile
import threading
import zipfile
from array import array
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from lxml import etree

from local_full_text_search.config.constants import TEMP_DIR
from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.parsers.base_parser import BaseParser
from local_full_text_search.parsers.ooxml.common import (
    clear_element,
    iterparse_end,
    safe_part_target,
    xml_root,
)

S_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
SHEET = f"{{{S_NS}}}sheet"
ROW = f"{{{S_NS}}}row"
CELL = f"{{{S_NS}}}c"
VALUE = f"{{{S_NS}}}v"
FORMULA = f"{{{S_NS}}}f"
TEXT = f"{{{S_NS}}}t"
SHARED_ITEM = f"{{{S_NS}}}si"
COMMENT = f"{{{S_NS}}}comment"
ROW_REF_RE = re.compile(r"(\d+)$")
CURSOR_SHIFT = 32
CURSOR_ROW_MASK = (1 << CURSOR_SHIFT) - 1


def encode_xlsx_cursor(sheet_index: int, row_number: int) -> int:
    return (max(0, int(sheet_index)) << CURSOR_SHIFT) | (
        max(0, int(row_number)) & CURSOR_ROW_MASK
    )


def decode_xlsx_cursor(cursor: int) -> tuple[int, int]:
    value = max(0, int(cursor))
    return value >> CURSOR_SHIFT, value & CURSOR_ROW_MASK


class SharedStringTable(Sequence[str]):
    """Compact shared-string storage with an mmap-backed large-file mode."""

    def __init__(
        self,
        *,
        values: list[str] | None = None,
        spool_path: Path | None = None,
        offsets: array[int] | None = None,
    ) -> None:
        self.values = values
        self.spool_path = spool_path
        self.offsets = offsets or array("Q")
        self.mode = "disk" if spool_path is not None else "memory"
        self._handle = None
        self._mapping: mmap.mmap | None = None
        if spool_path is not None and spool_path.stat().st_size:
            self._handle = spool_path.open("rb")
            self._mapping = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self) -> int:
        if self.values is not None:
            return len(self.values)
        return len(self.offsets) // 2

    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        normalized = int(index)
        if normalized < 0:
            normalized += len(self)
        if normalized < 0 or normalized >= len(self):
            raise IndexError(normalized)
        if self.values is not None:
            return self.values[normalized]
        if self._mapping is None:
            return ""
        start = int(self.offsets[normalized * 2])
        length = int(self.offsets[normalized * 2 + 1])
        return self._mapping[start : start + length].decode("utf-8")

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self.spool_path is not None:
            self.spool_path.unlink(missing_ok=True)


class XlsxStreamParser(BaseParser):
    name = "xlsx_stream"
    supports_resume = True

    def __init__(
        self,
        fallback: BaseParser | None = None,
        *,
        defer_normalization: bool = False,
        sheet_workers: int = 2,
        shared_strings_disk_threshold_bytes: int = 16 * 1024 * 1024,
        temp_dir: Path = TEMP_DIR,
    ) -> None:
        super().__init__()
        self.fallback = fallback
        self.defer_normalization = defer_normalization
        self.sheet_workers = max(1, min(4, int(sheet_workers)))
        self.shared_strings_disk_threshold_bytes = max(
            0,
            int(shared_strings_disk_threshold_bytes),
        )
        self.temp_dir = temp_dir
        self._progress_lock = threading.Lock()

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {".xlsx", ".xlsm"}

    def parse(self, file_path: Path, cancel_token: CancelToken) -> Iterable[ContentBlock]:
        try:
            yield from self._parse_stream(file_path, cancel_token)
        except CancelledError:
            raise
        except (
            OSError,
            zipfile.BadZipFile,
            KeyError,
            etree.XMLSyntaxError,
            ValueError,
            IndexError,
        ):
            if self.fallback is None:
                raise
            self.fallback.reset_status()
            yield from self.fallback.parse(file_path, cancel_token)
            if self.fallback.last_status != "success":
                self.set_status(
                    self.fallback.last_status,
                    self.fallback.last_error_code,
                    self.fallback.last_error_message,
                )

    def _parse_stream(
        self,
        file_path: Path,
        cancel_token: CancelToken,
    ) -> Iterable[ContentBlock]:
        with zipfile.ZipFile(file_path) as archive:
            available = set(archive.namelist())
            if (
                "xl/workbook.xml" not in available
                or "xl/_rels/workbook.xml.rels" not in available
            ):
                raise ValueError("XLSX is missing workbook parts")
            sheets = _sheet_parts(archive)
            max_sheet_bytes = max(
                (sheet_bytes for _, _, _, sheet_bytes in sheets),
                default=0,
            )
            self._report(
                "workbook_scan",
                completed=0,
                total=len(sheets),
                unit_type="sheet",
                detail=(
                    f"{len(sheets)} 个工作表 · "
                    f"最大 {max_sheet_bytes // 1024 // 1024} MB"
                ),
            )
            shared_strings = (
                _shared_strings(
                    archive,
                    cancel_token,
                    temp_dir=self.temp_dir,
                    disk_threshold_bytes=self.shared_strings_disk_threshold_bytes,
                )
                if "xl/sharedStrings.xml" in available
                else SharedStringTable(values=[])
            )
        try:
            if len(shared_strings):
                self._report(
                    "shared_strings",
                    completed=len(shared_strings),
                    total=len(shared_strings),
                    unit_type="string",
                    detail=(
                        f"共享字符串 {len(shared_strings):,} · "
                        f"{shared_strings.mode}"
                    ),
                )
            resume_sheet, resume_row = decode_xlsx_cursor(self.resume_cursor)
            use_parallel = (
                self.sheet_workers > 1
                and len(sheets) > 1
                and self.resume_cursor == 0
            )
            if use_parallel:
                yield from self._parse_parallel_sheets(
                    file_path,
                    sheets,
                    available,
                    shared_strings,
                    cancel_token,
                )
                return
            block_index = 0
            for sheet_index, sheet in enumerate(sheets, start=1):
                if sheet_index < resume_sheet:
                    continue
                row_cursor = resume_row if sheet_index == resume_sheet else 0
                for block in self._iter_sheet_blocks(
                    file_path,
                    sheet,
                    sheet_index,
                    len(sheets),
                    available,
                    shared_strings,
                    cancel_token,
                    resume_row=row_cursor,
                    sheet_parallel=False,
                ):
                    block.block_index = block_index
                    block_index += 1
                    yield block
        finally:
            shared_strings.close()

    def _parse_parallel_sheets(
        self,
        file_path: Path,
        sheets: list[tuple[str, str, int, int]],
        available: set[str],
        shared_strings: SharedStringTable,
        cancel_token: CancelToken,
    ) -> Iterable[ContentBlock]:
        worker_count = min(self.sheet_workers, len(sheets))
        executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="lfts-xlsx-sheet",
        )
        futures: dict[int, Future[list[ContentBlock]]] = {}
        next_submit = 1

        def submit(sheet_index: int) -> None:
            sheet = sheets[sheet_index - 1]
            futures[sheet_index] = executor.submit(
                lambda: list(
                    self._iter_sheet_blocks(
                        file_path,
                        sheet,
                        sheet_index,
                        len(sheets),
                        available,
                        shared_strings,
                        cancel_token,
                        resume_row=0,
                        sheet_parallel=True,
                    )
                )
            )

        for _ in range(worker_count):
            submit(next_submit)
            next_submit += 1
        block_index = 0
        try:
            for sheet_index in range(1, len(sheets) + 1):
                cancel_token.wait_if_paused()
                cancel_token.throw_if_cancelled()
                blocks = futures.pop(sheet_index).result()
                if next_submit <= len(sheets):
                    submit(next_submit)
                    next_submit += 1
                for block in blocks:
                    block.block_index = block_index
                    block_index += 1
                    yield block
        finally:
            executor.shutdown(
                wait=not cancel_token.cancelled,
                cancel_futures=True,
            )

    def _iter_sheet_blocks(
        self,
        file_path: Path,
        sheet: tuple[str, str, int, int],
        sheet_index: int,
        sheet_total: int,
        available: set[str],
        shared_strings: SharedStringTable,
        cancel_token: CancelToken,
        *,
        resume_row: int,
        sheet_parallel: bool,
    ) -> Iterator[ContentBlock]:
        sheet_name, sheet_part, row_estimate, sheet_bytes = sheet
        if sheet_part not in available:
            return
        self._report(
            "sheet_scan",
            completed=sheet_index,
            total=sheet_total,
            unit_type="sheet",
            cursor=encode_xlsx_cursor(sheet_index, resume_row),
            detail=f"{sheet_name} · {sheet_bytes // 1024} KB",
        )
        with zipfile.ZipFile(file_path) as archive:
            comments = _comments_for_sheet(
                archive,
                sheet_part,
                available,
                cancel_token,
            )
            row_reported = resume_row
            last_row_number = resume_row
            for row in iterparse_end(archive, sheet_part, ROW, cancel_token):
                row_number = int(row.get("r") or 0)
                if row_number:
                    last_row_number = row_number
                if row_number <= resume_row:
                    clear_element(row)
                    continue
                parts: list[str] = []
                first_cell: str | None = None
                last_cell: str | None = None
                for cell in row.iter(CELL):
                    coordinate = str(cell.get("r") or "")
                    value = _cell_text(cell, shared_strings)
                    if not coordinate or value is None:
                        continue
                    comment = comments.get(coordinate)
                    if comment:
                        value = f"{value}; 批注: {comment}"
                    first_cell = first_cell or coordinate
                    last_cell = coordinate
                    parts.append(f"{coordinate}={value}")
                if parts:
                    yield self.make_block(
                        file_path,
                        0,
                        "xlsx_row",
                        f"Sheet: {sheet_name}; 第 {row_number} 行",
                        " | ".join(parts),
                        sheet_name=sheet_name,
                        cell_start=first_cell,
                        cell_end=last_cell,
                        extra={
                            "row_start": row_number,
                            "row_end": row_number,
                            "sheet_parallel": sheet_parallel,
                            "shared_strings_mode": shared_strings.mode,
                        },
                    )
                if row_number and (
                    row_number - row_reported >= 250
                    or row_number == row_estimate
                ):
                    self._report(
                        "sheet_row",
                        completed=row_number,
                        total=row_estimate or row_number,
                        unit_type="row",
                        cursor=encode_xlsx_cursor(sheet_index, row_number),
                        detail=sheet_name,
                    )
                    row_reported = row_number
                clear_element(row)
            final_row = last_row_number or row_reported or row_estimate
            self._report(
                "sheet_row",
                completed=final_row,
                total=row_estimate or max(final_row, 1),
                unit_type="row",
                cursor=encode_xlsx_cursor(sheet_index, final_row),
                detail=f"{sheet_name} 完成",
            )

    def _report(
        self,
        phase: str,
        *,
        completed: int = 0,
        total: int = 0,
        unit_type: str = "",
        cursor: int | str | None = None,
        detail: str = "",
    ) -> None:
        with self._progress_lock:
            self.report_progress(
                phase,
                completed=completed,
                total=total,
                unit_type=unit_type,
                cursor=cursor,
                detail=detail,
            )


def _shared_strings(
    archive: zipfile.ZipFile,
    cancel_token: CancelToken,
    *,
    temp_dir: Path,
    disk_threshold_bytes: int,
) -> SharedStringTable:
    info = archive.getinfo("xl/sharedStrings.xml")
    use_disk = int(info.file_size) >= max(0, int(disk_threshold_bytes))
    if not use_disk:
        strings: list[str] = []
        for item in iterparse_end(
            archive,
            "xl/sharedStrings.xml",
            SHARED_ITEM,
            cancel_token,
        ):
            strings.append("".join(node.text or "" for node in item.iter(TEXT)))
            clear_element(item)
        return SharedStringTable(values=strings)

    temp_dir.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix="shared_strings_",
        suffix=".bin",
        dir=temp_dir,
        delete=False,
    )
    spool_path = Path(handle.name)
    offsets = array("Q")
    try:
        with handle:
            for item in iterparse_end(
                archive,
                "xl/sharedStrings.xml",
                SHARED_ITEM,
                cancel_token,
            ):
                encoded = "".join(
                    node.text or "" for node in item.iter(TEXT)
                ).encode("utf-8")
                start = handle.tell()
                handle.write(encoded)
                offsets.extend((start, len(encoded)))
                clear_element(item)
        if not offsets:
            spool_path.unlink(missing_ok=True)
            return SharedStringTable(values=[])
        return SharedStringTable(spool_path=spool_path, offsets=offsets)
    except Exception:
        spool_path.unlink(missing_ok=True)
        raise


def _sheet_parts(archive: zipfile.ZipFile) -> list[tuple[str, str, int, int]]:
    rels_root = xml_root(archive, "xl/_rels/workbook.xml.rels")
    targets: dict[str, str] = {}
    for relation in rels_root.iter(f"{{{REL_NS}}}Relationship"):
        relation_id = str(relation.get("Id") or "")
        if str(relation.get("Type") or "").endswith("/worksheet"):
            target = safe_part_target(
                "xl/workbook.xml",
                str(relation.get("Target") or ""),
            )
            if target:
                targets[relation_id] = target
    workbook_root = xml_root(archive, "xl/workbook.xml")
    result: list[tuple[str, str, int, int]] = []
    for sheet in workbook_root.iter(SHEET):
        name = str(sheet.get("name") or "Sheet")
        relation_id = str(sheet.get(f"{{{R_NS}}}id") or "")
        target = targets.get(relation_id)
        if target:
            row_estimate = _sheet_row_estimate(archive, target)
            try:
                sheet_bytes = int(archive.getinfo(target).file_size)
            except KeyError:
                sheet_bytes = 0
            result.append((name, target, row_estimate, sheet_bytes))
    return result


def _sheet_row_estimate(archive: zipfile.ZipFile, sheet_part: str) -> int:
    try:
        with archive.open(sheet_part) as source:
            context = etree.iterparse(
                source,
                events=("start",),
                tag=f"{{{S_NS}}}dimension",
                resolve_entities=False,
                no_network=True,
                recover=False,
                huge_tree=True,
            )
            for _, element in context:
                ref = str(element.get("ref") or "")
                if ref:
                    return _row_count_from_dimension(ref)
                break
    except Exception:
        return 0
    return 0


def _row_count_from_dimension(ref: str) -> int:
    if ":" not in ref:
        return 1 if ref else 0
    tail = ref.rsplit(":", 1)[-1]
    match = ROW_REF_RE.search(tail)
    return int(match.group(1)) if match else 0


def _comments_for_sheet(
    archive: zipfile.ZipFile,
    sheet_part: str,
    available: set[str],
    cancel_token: CancelToken,
) -> dict[str, str]:
    sheet_name = Path(sheet_part).name
    rels_part = f"xl/worksheets/_rels/{sheet_name}.rels"
    if rels_part not in available:
        return {}
    root = xml_root(archive, rels_part)
    comments_part: str | None = None
    for relation in root.iter(f"{{{REL_NS}}}Relationship"):
        if str(relation.get("Type") or "").endswith("/comments"):
            comments_part = safe_part_target(
                sheet_part,
                str(relation.get("Target") or ""),
            )
            break
    if not comments_part or comments_part not in available:
        return {}
    result: dict[str, str] = {}
    for comment in iterparse_end(
        archive,
        comments_part,
        COMMENT,
        cancel_token,
    ):
        ref = str(comment.get("ref") or "")
        text = "".join(node.text or "" for node in comment.iter(TEXT)).strip()
        if ref and text:
            result[ref] = text
        clear_element(comment)
    return result


def _cell_text(
    cell: etree._Element,
    shared_strings: Sequence[str],
) -> str | None:
    cell_type = str(cell.get("t") or "")
    formula = cell.find(FORMULA)
    if formula is not None and formula.text:
        return "=" + formula.text
    if cell_type == "inlineStr":
        text = "".join(node.text or "" for node in cell.iter(TEXT))
        return text if text else None
    value = cell.find(VALUE)
    if value is None or value.text is None:
        return None
    raw = value.text
    if cell_type == "s":
        return shared_strings[int(raw)]
    if cell_type == "b":
        return "TRUE" if raw == "1" else "FALSE"
    return raw
