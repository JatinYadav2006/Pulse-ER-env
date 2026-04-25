"""Central tool registry and validation helpers for Pulse-ER."""

from __future__ import annotations

from dataclasses import dataclass
import re
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


_NUMERIC_PREFIX_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)")


def _coerce_numeric_argument(value: Any) -> float:
    """Coerce model-emitted numeric strings like ``2LPM`` into floats.

    The policy layer sometimes emits recoverable formatting artifacts such as
    ``"2LPM"`` or ``"500 ml"``. These should be treated as parsing noise
    rather than hard clinical failures, as long as a leading numeric value is
    still obvious and unambiguous.
    """

    if isinstance(value, bool):
        raise TypeError("boolean values are not valid numeric tool arguments")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        compact = value.strip().replace(",", "")
        try:
            return float(compact)
        except ValueError:
            match = _NUMERIC_PREFIX_RE.match(compact)
            if match is not None:
                return float(match.group(1))
    raise ValueError("numeric coercion failed")


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
                numeric=True,
                minimum=0.1,
            ),
            ToolArgumentSpec(
                name="device",
                description="Optional oxygen delivery device such as nasal_cannula or non_rebreather_mask.",
            ),
            ToolArgumentSpec(
                name="monitor_seconds",
                description="Optional observation window after oxygen is applied.",
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
                numeric=True,
                minimum=1.0,
            ),
            ToolArgumentSpec(
                name="fluid_type",
                description="Optional fluid selection such as saline, blood, or packed_rbc.",
            ),
            ToolArgumentSpec(
                name="fluid",
                description="Alias for fluid_type when a shorter key is used.",
            ),
            ToolArgumentSpec(
                name="bag_volume_ml",
                description="Alias for total infused volume in milliliters.",
                numeric=True,
                minimum=1.0,
            ),
            ToolArgumentSpec(
                name="rate_ml_per_min",
                description="Infusion rate in milliliters per minute.",
                numeric=True,
                minimum=0.1,
            ),
            ToolArgumentSpec(
                name="monitor_seconds",
                description="Optional observation window after the infusion starts.",
                numeric=True,
                minimum=0.1,
            ),
        ),
    ),
    ToolSpec(
        tool_name="control_bleeding",
        tier="tier_2",
        description="Apply bleeding control measures for active hemorrhage.",
        read_only=False,
        state_changing=True,
        arguments=(
            ToolArgumentSpec(
                name="site",
                description="Optional hemorrhage site when more than one bleed is active.",
            ),
            ToolArgumentSpec(
                name="compartment",
                description="Alias for the hemorrhage site or compartment.",
            ),
            ToolArgumentSpec(
                name="method",
                description="Bleeding control technique to use.",
                choices=("tourniquet", "pressure", "hemostatic_dressing"),
            ),
            ToolArgumentSpec(
                name="monitor_seconds",
                description="Optional observation window after control is applied.",
                numeric=True,
                minimum=0.1,
            ),
        ),
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
                description="Target position when explicitly specified, such as supine or upright.",
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
                description="Optional airway support mode such as cpap, bag_valve_mask, or pressure_control_ventilation.",
            ),
            ToolArgumentSpec(
                name="support_type",
                description="Alias for the airway support mode or airway intervention type.",
            ),
            ToolArgumentSpec(
                name="monitor_seconds",
                description="Optional observation window after airway support is applied.",
                numeric=True,
                minimum=0.1,
            ),
            ToolArgumentSpec(
                name="fio2",
                description="Fraction of inspired oxygen as a 0-1 value.",
                numeric=True,
                minimum=0.0,
            ),
            ToolArgumentSpec(
                name="fraction_inspired_oxygen",
                description="Alias for fio2.",
                numeric=True,
                minimum=0.0,
            ),
            ToolArgumentSpec(
                name="peep_cmh2o",
                description="Positive end-expiratory pressure in cmH2O.",
                numeric=True,
            ),
            ToolArgumentSpec(
                name="peep",
                description="Alias for peep_cmh2o.",
                numeric=True,
            ),
            ToolArgumentSpec(
                name="respiration_rate_bpm",
                description="Ventilatory rate in breaths per minute.",
                numeric=True,
                minimum=1.0,
            ),
            ToolArgumentSpec(
                name="rate_bpm",
                description="Alias for respiration_rate_bpm.",
                numeric=True,
                minimum=1.0,
            ),
            ToolArgumentSpec(
                name="ie_ratio",
                description="Inspiratory-to-expiratory ratio.",
                numeric=True,
                minimum=0.1,
            ),
            ToolArgumentSpec(
                name="inspiratory_expiratory_ratio",
                description="Alias for ie_ratio.",
                numeric=True,
                minimum=0.1,
            ),
            ToolArgumentSpec(
                name="squeeze_pressure_cmh2o",
                description="Bag-valve-mask squeeze pressure in cmH2O.",
                numeric=True,
            ),
            ToolArgumentSpec(
                name="pressure_cmh2o",
                description="Alias for squeeze or inspiratory pressure in cmH2O.",
                numeric=True,
            ),
            ToolArgumentSpec(
                name="squeeze_volume_ml",
                description="Bag-valve-mask squeeze volume in milliliters.",
                numeric=True,
                minimum=1.0,
            ),
            ToolArgumentSpec(
                name="tidal_volume_ml",
                description="Alias for squeeze_volume_ml.",
                numeric=True,
                minimum=1.0,
            ),
            ToolArgumentSpec(
                name="airway_adjunct",
                description="Optional airway adjunct such as oropharyngeal or nasopharyngeal.",
            ),
            ToolArgumentSpec(
                name="pressure_support_cmh2o",
                description="Pressure support level in cmH2O.",
                numeric=True,
            ),
            ToolArgumentSpec(
                name="pressure_support",
                description="Alias for pressure_support_cmh2o.",
                numeric=True,
            ),
            ToolArgumentSpec(
                name="inspiratory_pressure_cmh2o",
                description="Inspiratory pressure target in cmH2O.",
                numeric=True,
            ),
            ToolArgumentSpec(
                name="inspiratory_period_s",
                description="Inspiratory period in seconds.",
                numeric=True,
                minimum=0.1,
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
    ToolSpec(
        tool_name="give_pressor",
        tier="tier_2",
        description="Start, titrate, or stop a vasopressor infusion for refractory hypotension.",
        read_only=False,
        state_changing=True,
        arguments=(
            ToolArgumentSpec(
                name="pressor",
                description="Pressor agent name such as norepinephrine or phenylephrine.",
            ),
            ToolArgumentSpec(
                name="agent",
                description="Alias for the pressor agent name.",
            ),
            ToolArgumentSpec(
                name="rate_ml_per_min",
                description="Infusion rate in milliliters per minute.",
                numeric=True,
                minimum=0.0,
            ),
            ToolArgumentSpec(
                name="concentration_ug_per_ml",
                description="Pressor concentration in micrograms per milliliter.",
                numeric=True,
                minimum=0.1,
            ),
            ToolArgumentSpec(
                name="stop",
                description="Set to true to stop the current infusion.",
            ),
            ToolArgumentSpec(
                name="monitor_seconds",
                description="Optional observation window after the pressor change.",
                numeric=True,
                minimum=0.1,
            ),
        ),
    ),
    ToolSpec(
        tool_name="needle_decompression",
        tier="tier_2",
        description="Perform needle decompression when a tension pneumothorax is suspected.",
        read_only=False,
        state_changing=True,
        arguments=(
            ToolArgumentSpec(
                name="side",
                description="Target side for decompression.",
                choices=("left", "right"),
            ),
            ToolArgumentSpec(
                name="monitor_seconds",
                description="Optional observation window after decompression.",
                numeric=True,
                minimum=0.1,
            ),
        ),
    ),
    ToolSpec(
        tool_name="pericardiocentesis",
        tier="tier_2",
        description="Perform pericardiocentesis when tamponade physiology is suspected.",
        read_only=False,
        state_changing=True,
        arguments=(
            ToolArgumentSpec(
                name="drain_rate_ml_per_min",
                description="Drainage rate in milliliters per minute.",
                numeric=True,
                minimum=0.1,
            ),
            ToolArgumentSpec(
                name="rate_ml_per_min",
                description="Alias for drain_rate_ml_per_min.",
                numeric=True,
                minimum=0.1,
            ),
            ToolArgumentSpec(
                name="monitor_seconds",
                description="Optional observation window after pericardiocentesis.",
                numeric=True,
                minimum=0.1,
            ),
        ),
    ),
    ToolSpec(
        tool_name="get_respiratory_status",
        tier="tier_1",
        description="Read a respiratory-focused bedside summary including breath sounds and EtCO2.",
        read_only=True,
        state_changing=False,
    ),
    ToolSpec(
        tool_name="get_blood_gas",
        tier="tier_1",
        description="Order or review arterial blood gas results.",
        read_only=True,
        state_changing=False,
    ),
    ToolSpec(
        tool_name="get_cbc",
        tier="tier_1",
        description="Order or review complete blood count results.",
        read_only=True,
        state_changing=False,
    ),
    ToolSpec(
        tool_name="get_bmp",
        tier="tier_1",
        description="Order or review basic metabolic panel results.",
        read_only=True,
        state_changing=False,
    ),
)

TOOL_SPEC_BY_NAME: dict[str, ToolSpec] = {
    spec.tool_name: spec for spec in TOOL_SPECS
}

INITIAL_TOOL_NAMES = [spec.tool_name for spec in TOOL_SPECS[:10]]
EXTENDED_TOOL_NAMES = [spec.tool_name for spec in TOOL_SPECS]
KNOWN_TOOL_NAMES = EXTENDED_TOOL_NAMES


def get_tool_spec(tool_name: str) -> ToolSpec:
    """Return the registry entry for one tool name."""

    try:
        return TOOL_SPEC_BY_NAME[tool_name]
    except KeyError as exc:
        raise ToolValidationError(f"Unsupported tool_name '{tool_name}'.") from exc


def build_tool_catalog(available_tools: list[str] | None = None) -> list[dict[str, Any]]:
    """Build a prompt-safe catalog of supported tools and arguments."""

    catalog: list[dict[str, Any]] = []
    for tool_name in available_tools or KNOWN_TOOL_NAMES:
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
                numeric_value = _coerce_numeric_argument(value)
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
