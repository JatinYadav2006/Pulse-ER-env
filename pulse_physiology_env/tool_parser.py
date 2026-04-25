"""Utilities for parsing model output into structured tool calls."""

from __future__ import annotations

import json
import re
import warnings
from typing import Any

from .models import ToolAction
from .tool_catalog import KNOWN_TOOL_NAMES, ToolValidationError, validate_tool_arguments


class ToolParseError(ValueError):
    """Raised when model output cannot be converted into a ToolAction."""


class ParseError(ToolParseError):
    """Raised when tool-call JSON cannot be extracted or validated safely."""


class ParseWarning(UserWarning):
    """Warning emitted when the parser must use a weaker extraction fallback."""


def _format_parse_error(message: str, raw_output: str) -> str:
    """Attach a compact raw-output preview to parser failures for debugging."""

    preview = raw_output.strip().replace("\n", "\\n")
    if len(preview) > 240:
        preview = preview[:237] + "..."
    return f"{message} Raw output: {preview}"


def _normalize_action_payload(payload: Any, raw_output: str) -> dict[str, Any]:
    """Normalize schema variants into a canonical tool-action payload."""

    if not isinstance(payload, dict):
        raise ParseError(_format_parse_error("Parsed JSON must be an object.", raw_output))

    if "action" in payload:
        nested_action = payload["action"]
        if not isinstance(nested_action, dict):
            raise ParseError(
                _format_parse_error("Top-level 'action' must itself be a JSON object.", raw_output)
            )
        if "reasoning" in payload and "reasoning" not in nested_action:
            nested_action = {**nested_action, "reasoning": payload["reasoning"]}
        payload = nested_action

    if "tool_name" not in payload:
        raise ParseError(
            _format_parse_error(
                "Tool-call JSON must contain either a top-level 'tool_name' or 'action' key.",
                raw_output,
            )
        )

    return payload


def _decode_json(candidate: str, raw_output: str) -> dict[str, Any]:
    """Decode one JSON candidate and normalize it to the expected action schema."""

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ParseError(
            _format_parse_error(f"Could not decode model output as JSON: {exc}", raw_output)
        ) from exc

    return _normalize_action_payload(payload, raw_output)


def _iter_schema_objects(text: str) -> list[dict[str, Any]]:
    """Scan raw text for standalone JSON objects with tool-action top-level keys."""

    decoder = json.JSONDecoder()
    matches: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _end_index = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and ("tool_name" in payload or "action" in payload):
            matches.append(payload)
    return matches


def parse_with_fallback(llm_output: str, log_warnings: bool = True) -> dict[str, Any]:
    """Parse LLM output with a strict extraction hierarchy and visible fallbacks."""

    candidate = llm_output.strip()
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    for block in fenced_blocks:
        try:
            return _decode_json(block.strip(), llm_output)
        except ParseError:
            continue

    for payload in _iter_schema_objects(candidate):
        return _normalize_action_payload(payload, llm_output)

    first = candidate.find("{")
    last = candidate.rfind("}")
    if first != -1 and last != -1 and last > first:
        if log_warnings:
            warnings.warn(
                "Parser fell back to broad brace extraction because no fenced block or schema-keyed JSON object was found.",
                ParseWarning,
                stacklevel=2,
            )
        return _decode_json(candidate[first : last + 1], llm_output)

    raise ParseError(_format_parse_error("No JSON object could be extracted from model output.", llm_output))


def extract_json_object(text: str) -> dict[str, Any]:
    """Backward-compatible wrapper around the stricter hierarchical parser."""

    return parse_with_fallback(text, log_warnings=True)


def parse_tool_action(
    text: str,
    *,
    allowed_tools: list[str] | None = None,
) -> ToolAction:
    """Parse raw model output into a validated ToolAction."""

    payload = parse_with_fallback(text, log_warnings=True)
    tool_name = payload.get("tool_name")
    arguments = payload.get("arguments", {})
    reasoning = payload.get("reasoning")

    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ParseError(_format_parse_error("tool_name must be a non-empty string.", text))

    if reasoning is not None and not isinstance(reasoning, str):
        raise ParseError(_format_parse_error("reasoning must be a string when provided.", text))

    valid_tools = allowed_tools or list(KNOWN_TOOL_NAMES)
    try:
        normalized_arguments = validate_tool_arguments(
            tool_name,
            arguments,
            allowed_tools=valid_tools,
        )
    except ToolValidationError as exc:
        raise ParseError(_format_parse_error(str(exc), text)) from exc

    return ToolAction(
        tool_name=tool_name,
        arguments=normalized_arguments,
        reasoning=reasoning,
    )
