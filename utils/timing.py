"""Utilities for measuring execution duration."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def track_duration() -> Generator[callable, None, None]:
    """Context manager that yields a callable returning elapsed milliseconds."""

    start = time.perf_counter()

    def _elapsed_ms() -> float:
        return (time.perf_counter() - start) * 1000

    yield _elapsed_ms
