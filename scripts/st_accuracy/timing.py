from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class TimingCollector:
    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._values: dict[str, float] = {}

    @contextmanager
    def measure(self, key: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(key, (time.perf_counter() - start) * 1000.0)

    def add(self, key: str, value_ms: float) -> None:
        self._values[key] = self._values.get(key, 0.0) + float(value_ms)

    def to_json_dict(self) -> dict[str, float]:
        values = dict(self._values)
        values["total_ms"] = (time.perf_counter() - self._start) * 1000.0
        return {key: round(value, 3) for key, value in sorted(values.items())}
