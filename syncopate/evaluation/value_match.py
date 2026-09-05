"""Helpers shared by teacher probes and SFT material generation."""

from __future__ import annotations

import json
import re


def gold_values(answer_text: str) -> list[str]:
    """Extract scalar values from the structured ``answer`` payload."""
    match = re.search(r"\{.*\}", answer_text, re.S)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    values: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif value is not None:
            text = str(value).strip()
            if text and text.lower() != "answered":
                values.append(text)

    walk(payload.get("answer", {}))
    return values
