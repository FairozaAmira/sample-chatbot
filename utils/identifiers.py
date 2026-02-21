"""Helpers for generating request identifiers."""
from __future__ import annotations

import uuid


def generate_request_id() -> str:
    """Return a unique request identifier."""

    return f"req_{uuid.uuid4().hex}"
