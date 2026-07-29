from __future__ import annotations


class AppError(Exception):
    """Base application exception."""


class CancelledError(AppError):
    """Raised when a long running task is cancelled."""


class ParserDependencyError(AppError):
    """Raised when a parser dependency is not installed."""


class UnsupportedFormatError(AppError):
    """Raised when no parser supports a file."""


class PasswordProtectedError(AppError):
    """Raised for encrypted documents without a password."""


class PartialParseError(AppError):
    """Raised when a parser can keep metadata but not extract body text."""


class IndexNotReadyError(AppError):
    """Raised when search is attempted before the complete index is published."""
