from __future__ import annotations


class AppError(Exception):
    """Base application exception."""


class CancelledError(AppError):
    """Raised when a long running task is cancelled."""


class PauseRequestedError(AppError):
    """Raised at a parser safe point so the scheduler can persist and pause."""


class ParserDependencyError(AppError):
    """Raised when a parser dependency is not installed."""


class UnsupportedFormatError(AppError):
    """Raised when no parser supports a file."""


class PasswordProtectedError(AppError):
    """Raised for encrypted documents without a password."""


class ZipMemberDirectoryChangedError(OSError):
    """Raised when a ZIP central-directory member no longer matches planning."""


class ZipMemberSizeChangedError(OSError):
    """Raised when a ZIP member size changes between planning and extraction."""


class ZipMemberContentChangedError(OSError):
    """Raised when a ZIP member's exact bytes change between planning and extraction."""


class ZipMemberEncryptedError(AppError):
    """Raised when a ZIP member requires a password."""


class PartialParseError(AppError):
    """Raised when a parser can keep metadata but not extract body text."""


class IndexNotReadyError(AppError):
    """Raised when search is attempted before the complete index is published."""


class PlanningWorkerError(AppError):
    """Raised when a recoverable planning worker fails."""


class PlanningNoProgressError(PlanningWorkerError):
    """Raised after a planning process stops making semantic progress."""
