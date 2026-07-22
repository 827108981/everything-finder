from __future__ import annotations

import json
from pathlib import Path

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings, DEFAULT_SETTINGS


class SettingsService:
    def __init__(self, path: Path = SETTINGS_PATH) -> None:
        self.path = path

    def load(self) -> AppSettings:
        if not self.path.exists():
            self.save(DEFAULT_SETTINGS)
            return DEFAULT_SETTINGS
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return AppSettings.from_dict(data)
        except (OSError, json.JSONDecodeError):
            pass
        return DEFAULT_SETTINGS

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(settings.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
