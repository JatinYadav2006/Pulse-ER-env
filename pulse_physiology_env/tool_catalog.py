"""Central tool registry and validation helpers for Pulse-ER."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolArgumentSpec:
    """Schema metadata for one structured tool argument."""

    name: str
    description: str
    required: bool = False
    numeric: bool = False
    minimum: float | None = None
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolSpec:
    """Single source of truth for a supported tool."""

    tool_name: str
    tier: str
    description: str
    read_only: bool
    state_changing: bool
    arguments: tuple[ToolArgumentSpec, ...] = ()


class ToolValidationError(ValueError):
    """Raised when a tool call payload violates the frozen contract."""


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        tool_name="get_vitals",
        tier="tier_1",
        description="Read the current core vitals without changing patient state.",
        read_only=True,
        state_changing=False,
    ),
    ToolSpec(
        tool_name="advance_time",
        tier="tier_2",
        description="Advance the simulation clock to observe ongoing physiology.",
        read_only=False,
        state_changing=True,
        arguments=(
            ToolArgumentSpec(
                name="seconds",
                description="Number of seconds to advance the simulation.",
                required=True,
                numeric=True,
                minimum=0.1,
            ),
        ),
    ),
    ToolSpec(
        tool_name="give_oxygen",
        tier="tier_2",
        description="Provide supplemental oxygen to improve oxygenation.",
        read_only=False,
        state_changing=True,
        arguments=(
            ToolArgumentSpec(
                name="flow_lpm",
                description="Oxygen flow rate in liters per minute.",
                required=True,
                numeric=True,
                minimum=0.1,
            ),
        ),
    ),
    ToolSpec(
        tool_name="give_fluids",
        tier="tier_2",
        description="Administer IV fluids to support perfusion and blood pressure.",
        read_only=False,
        state_changing=True,
        arguments=(
            ToolArgumentSpec(
                name="volume_ml",
                description="Fluid bolus volume in milliliters.",
                required=True,
                numeric=True,
                minimum=1.0,
            ),
        ),
    ),
    ToolSpec(
        tool_name="control_bleeding",
        tier="tier_2",
        description="Apply bleeding control measures for active hemorrhage.",
        read_only=False,
        state_changing=True,
    ),
    ToolSpec(
        tool_name="position_patient",
        tier="tier_2",
        description="Reposition the patient to support breathing or perfusion.",
        read_only=False,
        state_changing=True,
        arguments=(
            ToolArgumentSpec(
                name="position",
                description="Target position when explicitly specified.",
                choices=("supine", "upright", "left_lateral", "right_lateral"),
            ),
        ),
    ),
    ToolSpec(
        tool_name="airway_support",
        tier="tier_2",
        description="Provide airway support to improve ventilation.",
        read_only=False,
        state_changing=True,
        arguments=(
            ToolArgumentSpec(
                name="mode",
                description="Optional airway support mode.",
                choices=("basic", "advanced"),
            ),
        ),
    ),
    ToolSpec(
        tool_name="summarize_state",
        tier="tier_1",
        description="Summarize the current patient state in concise clinical language.",
        read_only=True,
        state_changing=False,
    ),
    ToolSpec(
        tool_name="check_deterioration",
        tier="tier_1",
        description="Assess whether the patient is currently worsening.",
        read_only=True,
        state_changing=False,
    ),
    ToolSpec(
        tool_name="recommend_next_step",
        tier="tier_3",
        description="Recommend the most appropriate next intervention or assessment.",
        read_only=True,
        state_changing=False,
    ),
)

TOOL_SPEC_BY_NAME: dict[str, ToolSpec] = {
    spec.tool_name: spec for spec in TOOL_SPECS
}

INITIAL_TOOL_NAMES = [spec.tool_name for spec in TOOL_SPECS]


def get_tool_spec(tool_name: str) -> ToolSpec:
    """Return the registry entry for one tool name."""

    try:
        return TOOL_SPEC_BY_NAME[tool_name]
    except KeyError as exc:
        raise ToolValidationError(f"Unsupported tool_name '{tool_name}'.") from exc


def build_tool_catalog(available_tools: list[str] | None = None) -> list[dict[str, Any]]:
    """Build a prompt-safe catalog of supported tools and arguments."""

    catalog: list[dict[str, Any]] = []
    for tool_name in available_tools or INITIAL_TOOL_NAMES:
        spec = get_tool_spec(tool_name)
        catalog.append(
            {
                "tool_name": spec.tool_name,
                "tier": spec.tier,
                "description": spec.description,
                "read_only": spec.read_only,
                "state_changing": spec.state_changing,
                "arguments": [
                    {
                        "name": arg.name,
                        "description": arg.description,
                        "required": arg.required,
                        "numeric": arg.numeric,
                        "minimum": arg.minimum,
                        "choices": list(arg.choices),
                    }
                    for arg in spec.arguments
                ],
            }
        )
    return catalog


def validate_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    allowed_tools: list[str] | None = None,
) -> dict[str, Any]:
    """Validate and normalize structured arguments for one tool call."""

    if allowed_tools is not None and tool_name not in allowed_tools:
        raise ToolValidationError(
            f"Unsupported tool_name '{tool_name}'. Expected one of: {', '.join(allowed_tools)}"
        )

    if not isinstance(arguments, dict):
        raise ToolValidationError("arguments must be a JSON object.")

    spec = get_tool_spec(tool_name)
    supported_args = {arg.name: arg for arg in spec.arguments}

    unknown_args = sorted(set(arguments) - set(supported_args))
    if unknown_args:
        raise ToolValidationError(
            f"{tool_name} received unsupported arguments: {', '.join(unknown_args)}"
        )

    normalized: dict[str, Any] = {}
    for arg_spec in spec.arguments:
        value = arguments.get(arg_spec.name)
        if value is None:
            if arg_spec.required:
                raise ToolValidationError(
                    f"{tool_name} requires argument '{arg_spec.name}'."
                )
            continue

        if arg_spec.numeric:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ToolValidationError(
                    f"{tool_name}.{arg_spec.name} must be numeric."
                ) from exc
            if arg_spec.minimum is not None and numeric_value < arg_spec.minimum:
                raise ToolValidationError(
                    f"{tool_name}.{arg_spec.name} must be >= {arg_spec.minimum}."
                )
            normalized[arg_spec.name] = numeric_value
            continue

        if arg_spec.choices:
            if not isinstance(value, str):
                raise ToolValidationError(
                    f"{tool_name}.{arg_spec.name} must be a string."
                )
            if value not in arg_spec.choices:
                raise ToolValidationError(
                    f"{tool_name}.{arg_spec.name} must be one of: {', '.join(arg_spec.choices)}"
                )
            normalized[arg_spec.name] = value
            continue

        normalized[arg_spec.name] = value

    return normalized
