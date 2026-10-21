from __future__ import annotations

from datetime import datetime


def run_name(prefix: str, _: str) -> str:
    return f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{prefix}"
