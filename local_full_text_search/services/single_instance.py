from __future__ import annotations

import getpass
import hashlib
import logging
import os
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket


LOGGER = logging.getLogger(__name__)


def default_server_name(app_id: str = "LocalFullTextSearch") -> str:
    identity = "|".join(
        (
            getpass.getuser(),
            os.environ.get("USERDOMAIN", ""),
            os.environ.get("LOCALAPPDATA", ""),
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{app_id}-{digest}"


class SingleInstanceGuard(QObject):
    """Own one per-user local endpoint and forward duplicate launches to the UI."""

    activation_requested = Signal()

    def __init__(self, server_name: str | None = None) -> None:
        super().__init__()
        self.server_name = server_name or default_server_name()
        self.server = QLocalServer(self)
        self.server.newConnection.connect(self._accept_connections)
        self.owns_server = False

    def acquire(self) -> bool:
        """Return True for the primary instance, False after notifying a primary."""

        if self._notify_existing():
            return False

        # A crashed process can leave a stale Unix socket. On Windows this is a no-op
        # for a live named pipe and clears only an unusable endpoint.
        QLocalServer.removeServer(self.server_name)
        if self.server.listen(self.server_name):
            self.owns_server = True
            LOGGER.info(
                "single-instance primary acquired pid=%s endpoint=%s",
                os.getpid(),
                self.server_name,
            )
            return True

        # Another process may have won the listen race after the first probe.
        if self._notify_existing(connect_timeout_ms=900):
            return False
        error = self.server.errorString() or "unknown local-server error"
        LOGGER.error(
            "single-instance activation failed pid=%s endpoint=%s error=%s",
            os.getpid(),
            self.server_name,
            error,
        )
        raise RuntimeError("程序已经运行，请在任务栏中打开")

    def close(self) -> None:
        if not self.owns_server:
            return
        self.server.close()
        QLocalServer.removeServer(self.server_name)
        self.owns_server = False
        LOGGER.info("single-instance primary released pid=%s", os.getpid())

    def _notify_existing(self, *, connect_timeout_ms: int = 350) -> bool:
        socket = QLocalSocket()
        socket.connectToServer(self.server_name)
        if not socket.waitForConnected(connect_timeout_ms):
            return False
        message = f"ACTIVATE {os.getpid()}\n".encode("ascii")
        socket.write(message)
        socket.flush()
        socket.waitForBytesWritten(350)
        existing_pid = "unknown"
        if socket.waitForReadyRead(500):
            response = bytes(socket.readAll()).decode("ascii", errors="replace").strip()
            if response.startswith("OK "):
                existing_pid = response.removeprefix("OK ").strip() or "unknown"
        socket.disconnectFromServer()
        LOGGER.info(
            "single-instance duplicate notified pid=%s existing_pid=%s endpoint=%s",
            os.getpid(),
            existing_pid,
            self.server_name,
        )
        return True

    def _accept_connections(self) -> None:
        while self.server.hasPendingConnections():
            socket = self.server.nextPendingConnection()
            if socket is None:
                continue
            if not socket.bytesAvailable():
                socket.waitForReadyRead(100)
            request = bytes(socket.readAll()).decode("ascii", errors="replace").strip()
            sender_pid = request.removeprefix("ACTIVATE ").strip() if request else "unknown"
            LOGGER.info(
                "single-instance activation received pid=%s sender_pid=%s",
                os.getpid(),
                sender_pid,
            )
            socket.write(f"OK {os.getpid()}\n".encode("ascii"))
            socket.flush()
            socket.waitForBytesWritten(100)
            socket.disconnectFromServer()
            self.activation_requested.emit()


def activate_window(window: object, *, log: Callable[[str], None] | None = None) -> None:
    """Restore and focus the existing top-level window from the Qt UI thread."""

    try:
        is_minimized = getattr(window, "isMinimized")
        if is_minimized():
            getattr(window, "showNormal")()
        else:
            getattr(window, "show")()
        getattr(window, "raise_")()
        getattr(window, "activateWindow")()
        if log is not None:
            log("已有窗口已恢复并请求置前")
        LOGGER.info("single-instance window activation requested pid=%s", os.getpid())
    except Exception:
        LOGGER.exception("single-instance window activation failed pid=%s", os.getpid())
        if log is not None:
            log("已有窗口激活失败，请在任务栏中打开")

