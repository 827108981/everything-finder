from __future__ import annotations

import threading
import os
from pathlib import Path
from typing import TYPE_CHECKING

from local_full_text_search.config.constants import CACHE_DIR
from local_full_text_search.config.defaults import AppSettings
if TYPE_CHECKING:
    from local_full_text_search.rag.embeddings.fastembed_local import FastEmbedLocalProvider


class RagModelManager:
    """Lazily downloads and reuses the configured local embedding model."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._embedding_key: tuple[str, str] | None = None
        self._embedding: FastEmbedLocalProvider | None = None

    def embedding_provider(self, settings: AppSettings) -> FastEmbedLocalProvider:
        from local_full_text_search.rag.embeddings.fastembed_local import FastEmbedLocalProvider

        model_id = str(settings.rag_embedding_model_id or "BAAI/bge-small-zh-v1.5").strip()
        cache_dir = (CACHE_DIR / "rag-fastembed").resolve()
        key = (model_id, str(cache_dir))
        with self._lock:
            if self._embedding is None or self._embedding_key != key:
                self._embedding = FastEmbedLocalProvider(
                    model_name=model_id,
                    cache_dir=cache_dir,
                    threads=max(1, min(8, int(settings.parser_workers))),
                )
                self._embedding_key = key
            return self._embedding

    def reset(self) -> None:
        with self._lock:
            self._embedding = None
            self._embedding_key = None

    def stop(self) -> None:
        self.reset()


class CredentialStore:
    """Store the LLM API key through the operating system credential store."""

    SERVICE = "LocalFullTextSearch.RAG"
    USERNAME = "llm_api_key"

    def __init__(self, keyring_module=None) -> None:
        self._keyring_module = keyring_module

    def _keyring(self):
        if self._keyring_module is not None:
            return self._keyring_module
        try:
            import keyring
        except ImportError as exc:
            raise RuntimeError("缺少 keyring 依赖，请按 requirements-rag.txt 安装") from exc
        backend = keyring.get_keyring()
        if os.name == "nt" and "Windows" not in type(backend).__module__:
            raise RuntimeError("Windows Credential Manager 后端不可用，拒绝使用明文凭据后端")
        return keyring

    def get_api_key(self) -> str:
        try:
            return str(self._keyring().get_password(self.SERVICE, self.USERNAME) or "")
        except Exception as exc:
            raise RuntimeError(f"无法读取 Windows 凭据库：{exc}") from exc

    def save_api_key(self, api_key: str) -> None:
        value = str(api_key or "").strip()
        if not value:
            raise ValueError("API key 不能为空")
        try:
            self._keyring().set_password(self.SERVICE, self.USERNAME, value)
        except Exception as exc:
            raise RuntimeError(f"无法保存到 Windows 凭据库：{exc}") from exc

    def delete_api_key(self) -> None:
        keyring = self._keyring()
        try:
            keyring.delete_password(self.SERVICE, self.USERNAME)
        except keyring.errors.PasswordDeleteError:
            return
        except Exception as exc:
            raise RuntimeError(f"无法删除 Windows 凭据：{exc}") from exc
