from __future__ import annotations

import json
import os
from typing import List

from app.config.settings import settings


def _store_path() -> str:
    if os.path.isabs(settings.bm25_store_path):
        return settings.bm25_store_path
    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(app_dir, settings.bm25_store_path)


def save_corpus(texts: List[str]) -> str:
    path = _store_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(texts, handle, ensure_ascii=False)
    return path


def load_corpus() -> List[str]:
    path = _store_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return [str(item) for item in data]
    except (OSError, json.JSONDecodeError):
        return []
    return []
