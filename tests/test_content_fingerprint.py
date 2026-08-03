from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from local_full_text_search.core.content_fingerprint import fingerprint_file_with_spool


def test_full_hash_fingerprint_spools_the_same_single_source_read() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        source = base / "source.txt"
        payload = b"SINGLE_SOURCE_READ_PAYLOAD"
        source.write_bytes(payload)

        fingerprint, spool_path, bytes_read = fingerprint_file_with_spool(
            source,
            base / "spool",
        )

        assert fingerprint.key == f"sha256:{hashlib.sha256(payload).hexdigest()}"
        assert bytes_read == len(payload)
        assert spool_path is not None
        assert spool_path.read_bytes() == payload


def test_ooxml_candidate_fingerprint_does_not_create_a_full_copy() -> None:
    import zipfile

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        source = base / "source.docx"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("word/document.xml", "<document>content</document>")

        fingerprint, spool_path, bytes_read = fingerprint_file_with_spool(
            source,
            base / "spool",
        )

        assert fingerprint.method == "zip_directory"
        assert spool_path is None
        assert bytes_read == 0
