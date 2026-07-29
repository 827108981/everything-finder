from __future__ import annotations

from pathlib import Path
from typing import Iterable

from local_full_text_search.config.constants import VIDEO_EXTENSIONS
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.parsers.base_parser import BaseParser


class MetadataOnlyParser(BaseParser):
    """Parser for files whose content is intentionally not decoded in this version.

    Video files can be important search targets by filename, but decoding audio
    or subtitles is outside this stage. Marking them metadata_only keeps the
    index useful without polluting the failure list.
    """

    name = "metadata_only"

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in VIDEO_EXTENSIONS

    def parse(self, file_path: Path, cancel_token: CancelToken) -> Iterable[ContentBlock]:
        cancel_token.throw_if_cancelled()
        self.set_status("metadata_only", "METADATA_ONLY", "该格式当前仅索引文件名和路径")
        return []
