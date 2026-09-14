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


class IndexQuiescedForSearch(AppError):
    """Index writes are intentionally deferred while a foreground search runs."""


class SearchLeaseBusy(AppError):
    """A foreground search could not obtain the process-local search lease."""


class QuiesceTimeout(SearchLeaseBusy):
    """Indexing did not reach a safe pause point before the search deadline."""


class SearchTimeout(AppError):
    """A search exceeded its query deadline."""


class UserCancelled(AppError):
    """A user explicitly cancelled a search."""


class SQLiteBusy(AppError):
    """A real SQLite busy/locked condition exhausted its bounded retry budget."""


class SnapshotTimeout(AppError):
    """Detailed readiness statistics exceeded their watchdog budget."""


class OperationCancelled(AppError):
    """An operation was cancelled by the user or shutdown supervisor."""


class PlanningWorkerError(AppError):
    """Raised when a recoverable planning worker fails."""


class PlanningNoProgressError(PlanningWorkerError):
    """Raised after a planning process stops making semantic progress."""
