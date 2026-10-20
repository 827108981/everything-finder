from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from local_full_text_search.rag.models import VectorHit, VectorRecord


class VectorStore(Protocol):
    def upsert(self, records: Sequence[VectorRecord]) -> None: ...

    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        *,
        root_ids: Sequence[int] = (),
        extensions: Sequence[str] = (),
        include_ocr: bool = True,
    ) -> list[VectorHit]: ...

    def count(self) -> int: ...

    def close(self) -> None: ...
