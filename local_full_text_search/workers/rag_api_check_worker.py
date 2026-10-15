from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.rag.model_manager import CredentialStore


class RagApiCheckWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    ended = Signal()

    def __init__(self, base_url: str, model: str) -> None:
        super().__init__()
        self.base_url = str(base_url)
        self.model = str(model)
        self._cancelled = threading.Event()
        self._gateway = None

    @Slot()
    def run(self) -> None:
        gateway = None
        try:
            from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway

            if self._cancelled.is_set():
                return
            api_key = CredentialStore().get_api_key()
            gateway = OpenAICompatibleGateway(
                self.base_url,
                api_key,
                self.model,
                connect_timeout=2.0,
                read_timeout=2.0,
            )
            self._gateway = gateway
            self.finished.emit(gateway.test_connection())
        except Exception as exc:
            self.failed.emit(str(exc) or exc.__class__.__name__)
        finally:
            if gateway is not None:
                gateway.close()
            self._gateway = None
            self.ended.emit()

    @Slot()
    def cancel(self) -> None:
        self._cancelled.set()
        gateway = self._gateway
        if gateway is not None:
            gateway.close()
