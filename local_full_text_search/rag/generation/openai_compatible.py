from __future__ import annotations

import json
import re
import threading
from urllib.parse import urlparse

import httpx

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.generation.base import LLMUnavailableError, ModelHealth


class OpenAICompatibleGateway:
    """Chat Completions gateway with user-configured URL, API key, and model."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        connect_timeout: float = 15.0,
        read_timeout: float = 180.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        if not self.api_key:
            raise LLMUnavailableError("请先在设置中保存 API key")
        if not self.model:
            raise LLMUnavailableError("请先在设置中填写 API model")
        self.timeout = httpx.Timeout(read_timeout, connect=connect_timeout)
        self._client = httpx.Client(timeout=self.timeout, transport=transport)
        self._response_lock = threading.Lock()
        self._active_response: httpx.Response | None = None

    def health(self) -> ModelHealth:
        if not self.api_key:
            return ModelHealth(False, self.model, "API key 未设置")
        if not self.model:
            return ModelHealth(False, message="API model 未设置")
        return ModelHealth(True, self.model, "API 配置已完成")

    def test_connection(self) -> ModelHealth:
        health = self.health()
        if not health.available:
            return health
        try:
            response = self._client.get(_models_endpoint(self.base_url), headers=self._headers())
            if response.status_code in {404, 405}:
                return ModelHealth(True, self.model, "配置完整；此服务不提供 /models 探测")
            response.raise_for_status()
            ids = {
                str(item.get("id") or "")
                for item in (response.json().get("data") or [])
                if isinstance(item, dict)
            }
            if ids and self.model not in ids:
                return ModelHealth(False, self.model, "API 可访问，但 models 列表中没有所选模型")
            return ModelHealth(True, self.model, "API 连接正常")
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            return ModelHealth(False, self.model, f"API 连接失败：{_safe_error(exc, self.api_key)}")

    def count_chat_tokens(self, messages: list[dict[str, str]]) -> int:
        try:
            import tiktoken
        except ImportError:
            return _conservative_char_count(messages)
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        count = 3 + 4 * len(messages)
        for message in messages:
            count += len(encoding.encode(str(message.get("role") or "")))
            count += len(encoding.encode(str(message.get("content") or "")))
        return count

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        cancel_token: CancelToken,
    ) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            # OpenAI has deprecated max_tokens and does not accept it for
            # reasoning models; max_completion_tokens is the current field.
            "max_completion_tokens": max(64, int(max_tokens)),
            "response_format": {"type": "json_object"},
        }
        chunks: list[str] = []
        try:
            with self._client.stream(
                "POST",
                _chat_endpoint(self.base_url),
                headers=self._headers(),
                json=body,
            ) as response:
                with self._response_lock:
                    self._active_response = response
                if response.status_code >= 400:
                    body_text = response.read().decode("utf-8", errors="replace")[:1000]
                    raise LLMUnavailableError(
                        f"API 返回 HTTP {response.status_code}: {_safe_error(body_text, self.api_key)}"
                    )
                for line in response.iter_lines():
                    cancel_token.throw_if_cancelled()
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    event = json.loads(payload)
                    choice = (event.get("choices") or [{}])[0]
                    content = (choice.get("delta") or {}).get("content")
                    if isinstance(content, str):
                        chunks.append(content)
                with self._response_lock:
                    self._active_response = None
        except LLMUnavailableError:
            raise
        except httpx.HTTPError as exc:
            if cancel_token.cancelled:
                cancel_token.throw_if_cancelled()
            raise LLMUnavailableError(f"API 请求失败：{_safe_error(exc, self.api_key)}") from exc
        except (json.JSONDecodeError, IndexError, TypeError, KeyError) as exc:
            raise LLMUnavailableError("API 返回了无法解析的 streaming 响应") from exc
        answer = "".join(chunks).strip()
        if not answer:
            raise LLMUnavailableError("API 返回空答案")
        return answer

    def close(self) -> None:
        self.cancel()
        self._client.close()

    def cancel(self) -> None:
        with self._response_lock:
            response = self._active_response
            self._active_response = None
        if response is not None:
            response.close()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


def _normalize_base_url(value: str) -> str:
    raw = str(value or "").strip().rstrip("/")
    parsed = urlparse(raw)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError("API URL 必须是完整的 http(s) 地址")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("API URL 中不能包含账号、密钥、query 或 fragment；API key 请单独保存")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("远程 API URL 必须使用 HTTPS")
    if raw.endswith("/chat/completions"):
        return raw
    if raw.endswith("/v1"):
        return raw
    return raw + "/v1"


def _chat_endpoint(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        return base_url
    return base_url + "/chat/completions"


def _models_endpoint(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]
    return base_url + "/models"


def _safe_error(value: object, api_key: str) -> str:
    text = str(value or "").replace(api_key, "[redacted]") if api_key else str(value or "")
    return re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+", r"\1[redacted]", text)[:1000]


def _conservative_char_count(messages: list[dict[str, str]]) -> int:
    return 16 + sum(len(str(message.get("content") or "")) for message in messages)
