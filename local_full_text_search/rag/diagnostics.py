from __future__ import annotations

import logging
import re
import traceback
from uuid import uuid4
from urllib.parse import urlsplit

from local_full_text_search.services.logging_service import log_event


_BEARER_PATTERN = re.compile(
    r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;\]}>\"']+"
)
_KEY_VALUE_PATTERN = re.compile(
    r"(?i)((?:api[_-]?key|access[_-]?token)\s*[:=]\s*)[^\s,;\]}>\"']+"
)
_URL_SECRET_PATTERN = re.compile(
    r"(?i)([?&](?:api[_-]?key|key|token|access[_-]?token)=)[^&#\s]+"
)


def new_rag_request_id(operation: str) -> str:
    safe_operation = re.sub(r"[^a-z0-9_-]+", "-", str(operation).lower()).strip("-")
    return f"rag-{safe_operation or 'task'}-{uuid4().hex[:12]}"


def provider_host(api_base_url: str) -> str:
    try:
        return urlsplit(str(api_base_url).strip()).hostname or ""
    except ValueError:
        return ""


def redact_sensitive_text(value: object, secrets: tuple[str, ...] | list[str] = ()) -> str:
    text = str(value)
    for secret in secrets:
        candidate = str(secret or "")
        if candidate:
            text = text.replace(candidate, "<redacted>")
    text = _BEARER_PATTERN.sub(r"\1<redacted>", text)
    text = _KEY_VALUE_PATTERN.sub(r"\1<redacted>", text)
    return _URL_SECRET_PATTERN.sub(r"\1<redacted>", text)


def safe_exception_report(
    exc: BaseException,
    *,
    secrets: tuple[str, ...] | list[str] = (),
) -> tuple[str, str]:
    try:
        stacktrace = "".join(traceback.format_exception(exc))
    except Exception:
        stacktrace = f"{type(exc).__name__}: {exc}"
    return (
        redact_sensitive_text(str(exc) or type(exc).__name__, secrets),
        redact_sensitive_text(stacktrace, secrets)[-12_000:],
    )


def log_rag_exception(
    event: str,
    exc: BaseException,
    *,
    request_id: str,
    component: str,
    secrets: tuple[str, ...] | list[str] = (),
    **fields: object,
) -> tuple[str, str]:
    message, stacktrace = safe_exception_report(exc, secrets=secrets)
    status_code = None
    provider_request_id = None
    provider_error_type = None
    provider_error_code = None
    root_error_type = None
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        root_error_type = type(current).__name__
        response = getattr(current, "response", None)
        if status_code is None:
            status_code = getattr(current, "status_code", None) or getattr(
                response, "status_code", None
            )
        headers = getattr(response, "headers", None)
        if provider_request_id is None and headers is not None:
            try:
                provider_request_id = headers.get("x-request-id") or headers.get("request-id")
            except Exception:
                provider_request_id = None
        body = getattr(current, "body", None)
        if isinstance(body, dict):
            error = body.get("error", body)
            if isinstance(error, dict):
                if provider_error_type is None and error.get("type") is not None:
                    provider_error_type = str(error["type"])
                if provider_error_code is None and error.get("code") is not None:
                    provider_error_code = str(error["code"])
        current = current.__cause__ or current.__context__

    log_event(
        event,
        level=logging.ERROR,
        component=component,
        request_id=request_id,
        error_type=type(exc).__name__,
        error_message=message[:2_000],
        stacktrace=stacktrace,
        http_status=status_code,
        provider_request_id=provider_request_id,
        provider_error_type=provider_error_type,
        provider_error_code=provider_error_code,
        root_error_type=root_error_type,
        **fields,
    )
    return message, stacktrace
