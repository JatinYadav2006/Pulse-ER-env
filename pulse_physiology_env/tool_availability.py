"""Consumer-side validation for backend-exposed tool availability.

This module keeps Person 2's prompt and policy layers fail-closed. The mock and
real backends may evolve independently during integration, so we validate the
adapter contract before building prompts or selecting actions.
"""

from __future__ import annotations

from .tool_catalog import TOOL_SPEC_BY_NAME


class ToolAvailabilityError(ValueError):
    """Raised when the backend omits or corrupts the available-tools contract."""


def validate_tool_availability(available_tools: list[str] | None) -> list[str]:
    """Validate adapter-exposed tools and return a normalized list.

    The consumer stack must fail closed here: a missing or empty tool list is a
    contract violation during integration, not a signal to expose the full
    catalog speculatively.
    """

    if available_tools is None:
        raise ToolAvailabilityError(
            "available_tools is missing from the backend contract; refusing to build prompts or select actions."
        )

    if not isinstance(available_tools, list):
        raise ToolAvailabilityError(
            f"available_tools must be a list of tool names, received {type(available_tools).__name__}."
        )

    normalized = [tool_name for tool_name in available_tools if isinstance(tool_name, str) and tool_name.strip()]
    if not normalized:
        raise ToolAvailabilityError(
            "available_tools is empty; the backend must expose at least one supported tool."
        )

    unknown_tools = sorted(set(normalized) - set(TOOL_SPEC_BY_NAME))
    if unknown_tools:
        raise ToolAvailabilityError(
            f"available_tools contains unknown tool names: {', '.join(unknown_tools)}."
        )

    return normalized
