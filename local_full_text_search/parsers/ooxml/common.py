from __future__ import annotations

import posixpath
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path

from lxml import etree

from local_full_text_search.core.task_manager import CancelToken


def xml_parts(archive: zipfile.ZipFile, pattern: str) -> list[str]:
    expression = re.compile(pattern)
    return sorted(
        (name for name in archive.namelist() if expression.fullmatch(name)),
        key=natural_part_key,
    )


def natural_part_key(name: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def safe_part_target(base_part: str, target: str) -> str | None:
    if target.startswith("/"):
        candidate = target.lstrip("/")
    else:
        candidate = posixpath.normpath(posixpath.join(posixpath.dirname(base_part), target))
    if candidate == ".." or candidate.startswith("../"):
        return None
    return candidate


def iterparse_end(
    archive: zipfile.ZipFile,
    part_name: str,
    tags: str | tuple[str, ...],
    cancel_token: CancelToken,
) -> Iterable[etree._Element]:
    with archive.open(part_name) as source:
        context = etree.iterparse(
            source,
            events=("end",),
            tag=tags,
            resolve_entities=False,
            no_network=True,
            recover=False,
            huge_tree=True,
        )
        for _, element in context:
            cancel_token.wait_if_paused()
            cancel_token.throw_if_cancelled()
            yield element


def clear_element(element: etree._Element) -> None:
    element.clear()
    parent = element.getparent()
    if parent is not None:
        while element.getprevious() is not None:
            del parent[0]


def xml_root(archive: zipfile.ZipFile, part_name: str) -> etree._Element:
    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False, huge_tree=True)
    with archive.open(part_name) as source:
        return etree.parse(source, parser).getroot()


def require_zip(path: Path) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)
