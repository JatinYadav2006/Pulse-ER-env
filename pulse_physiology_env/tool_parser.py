"""Utilities for parsing model output into structured tool calls."""

from __future__ import annotations

import json
import re
from typing import Any

from .models import ToolAction
from .tool_catalog import INITIAL_TOOL_NAMES, ToolValidationError, validate_tool_arguments


class ToolParseError(ValueError):
    """Raised when model output cannot be converted into a ToolAction."""


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from raw text or a fenced block."""

    candidate = text.strip()

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
    else:
        first = candidate.find("{")
        last = candidate.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidate = candidate[first : last + 1]

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ToolParseError(f"Could not decode model output as JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ToolParseError("Parsed JSON must be an object.")

    return payload


def parse_tool_action(
    text: str,
    *,
    allowed_tools: list[str] | None = None,
) -> ToolAction:
    """Parse raw model output into a validated ToolAction."""

    payload = extract_json_object(text)
    tool_name = payload.get("tool_name")
    arguments = payload.get("arguments", {})
    reasoning = payload.get("reasoning")

    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ToolParseError("tool_name must be a non-empty string.")

    if reasoning is not None and not isinstance(reasoning, str):
        raise ToolParseError("reasoning must be a string when provided.")

    valid_tools = allowed_tools or list(INITIAL_TOOL_NAMES)
    try:
        normalized_arguments = validate_tool_arguments(
            tool_name,
            arguments,
            allowed_tools=valid_tools,
        )
    except ToolValidationError as exc:
        raise ToolParseError(str(exc)) from exc

    return ToolAction(
        tool_name=tool_name,
        arguments=normalized_arguments,
        reasoning=reasoning,
    )
