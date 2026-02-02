from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env() -> None:
    candidates = [
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    loaded = False
    for path in candidates:
        if path.exists():
            load_dotenv(path)
            loaded = True
    if not loaded:
        load_dotenv()

    os.environ.setdefault("GRID_API_AUTH_MODE", "auto")
    os.environ.setdefault("GRID_API_AUTH_HEADER", "x-api-key")
    os.environ.setdefault("GRID_API_TOKEN_PREFIX", "")
