from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Stopwatch:
    elapsed_ms: float = 0.0

    @contextmanager
    def measure(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.elapsed_ms = (time.perf_counter() - start) * 1000.0
