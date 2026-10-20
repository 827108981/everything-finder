from __future__ import annotations

from typing import Protocol

from local_full_text_search.rag.errors import RagConfigurationError


class SecretStore(Protocol):
    def save(self, secret_ref: str, value: str) -> None: ...

    def retrieve(self, secret_ref: str) -> str | None: ...

    def delete(self, secret_ref: str) -> None: ...


class WindowsCredentialSecretStore:
    """Uses Windows Credential Manager; API keys are never written to settings."""

    def save(self, secret_ref: str, value: str) -> None:
        if not value:
            self.delete(secret_ref)
            return
        try:
            import win32cred
        except ImportError as exc:
            raise RagConfigurationError("当前系统不支持 Windows Credential Manager") from exc
        win32cred.CredWrite(
            {
                "Type": win32cred.CRED_TYPE_GENERIC,
                "TargetName": str(secret_ref),
                "UserName": "LocalFullTextSearch",
                # pywin32 接口在这里接收 Python 字符串，并自行转换为 Windows
                # CredentialBlob 所需的 UTF-16；提前 encode 会触发 bytes/Unicode 类型错误。
                "CredentialBlob": str(value),
                "Persist": win32cred.CRED_PERSIST_LOCAL_MACHINE,
            },
            0,
        )

    def retrieve(self, secret_ref: str) -> str | None:
        try:
            import win32cred
        except ImportError as exc:
            raise RagConfigurationError("当前系统不支持 Windows Credential Manager") from exc
        try:
            credential = win32cred.CredRead(str(secret_ref), win32cred.CRED_TYPE_GENERIC)
        except Exception as exc:
            error_code = getattr(exc, "winerror", None)
            if error_code is None and getattr(exc, "args", None):
                first_arg = exc.args[0]
                error_code = first_arg if isinstance(first_arg, int) else None
            if error_code == 1168:  # ERROR_NOT_FOUND：尚无已保存的 Key。
                return None
            raise RagConfigurationError(
                f"无法读取 Windows 凭据（错误码 {error_code or type(exc).__name__}）"
            ) from exc
        blob = credential.get("CredentialBlob")
        if isinstance(blob, bytes):
            return blob.decode("utf-16-le")
        return str(blob) if blob else None

    def delete(self, secret_ref: str) -> None:
        try:
            import win32cred
        except ImportError as exc:
            raise RagConfigurationError("当前系统不支持 Windows Credential Manager") from exc
        try:
            win32cred.CredDelete(str(secret_ref), win32cred.CRED_TYPE_GENERIC, 0)
        except Exception as exc:
            error_code = getattr(exc, "winerror", None)
            if error_code is None and getattr(exc, "args", None):
                first_arg = exc.args[0]
                error_code = first_arg if isinstance(first_arg, int) else None
            if error_code == 1168:  # 删除不存在的凭据视为已达到目标状态。
                return
            raise RagConfigurationError(
                f"无法删除 Windows 凭据（错误码 {error_code or type(exc).__name__}）"
            ) from exc


class MemorySecretStore:
    """Test-only in-memory credential provider."""

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def save(self, secret_ref: str, value: str) -> None:
        self._values[str(secret_ref)] = str(value)

    def retrieve(self, secret_ref: str) -> str | None:
        return self._values.get(str(secret_ref))

    def delete(self, secret_ref: str) -> None:
        self._values.pop(str(secret_ref), None)
