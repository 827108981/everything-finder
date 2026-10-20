from __future__ import annotations

import sys
from types import SimpleNamespace

from local_full_text_search.rag.secrets import WindowsCredentialSecretStore


def test_windows_credential_blob_is_passed_as_unicode(monkeypatch) -> None:
    captured = {}
    fake_credential = SimpleNamespace(
        CRED_TYPE_GENERIC=1,
        CRED_PERSIST_LOCAL_MACHINE=2,
        CredWrite=lambda credential, flags: captured.update(
            credential=credential, flags=flags
        ),
    )
    monkeypatch.setitem(sys.modules, "win32cred", fake_credential)

    WindowsCredentialSecretStore().save("test-target", "test-api-key")

    credential = captured["credential"]
    assert isinstance(credential["CredentialBlob"], str)
    assert credential["CredentialBlob"] == "test-api-key"


def test_windows_credential_read_decodes_utf16_blob(monkeypatch) -> None:
    fake_credential = SimpleNamespace(
        CRED_TYPE_GENERIC=1,
        CredRead=lambda target, credential_type: {
            "CredentialBlob": "test-api-key".encode("utf-16-le")
        },
    )
    monkeypatch.setitem(sys.modules, "win32cred", fake_credential)

    assert WindowsCredentialSecretStore().retrieve("test-target") == "test-api-key"
