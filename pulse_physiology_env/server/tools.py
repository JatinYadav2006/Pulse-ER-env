"""Tool registry and execution logic for the Pulse physiology environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pulse_physiology_env.models import PulsePhysiologyAction, ToolError, ToolResult
from pulse_physiology_env.patient_state import PatientState

from .pulse_engine_adapter import PulseEngineAdapter

INITIAL_TOOL_NAMES = [
    "get_vitals",
    "advance_time",
    "give_oxygen",
    "give_fluids",
    "control_bleeding",
    "position_patient",
    "airway_support",
    "summarize_state",
    "check_deterioration",
    "recommend_next_step",
]

EXTENDED_TOOL_NAMES = INITIAL_TOOL_NAMES + [
    "needle_decompression",
    "get_respiratory_status",
    "get_blood_gas",
    "get_cbc",
    "get_bmp",
]


@dataclass
class ToolExecution:
    """Outcome of a handled tool call."""

    state: PatientState
    tool_result: ToolResult
    error: ToolError | None


class PulseToolExecutor:
    """Executes named tools against a PulseEngineAdapter."""

    def __init__(self, adapter: PulseEngineAdapter) -> None:
        self._adapter = adapter
        self._handlers: dict[str, Callable[[dict[str, Any]], ToolExecution]] = {
            "get_vitals": self._handle_get_vitals,
            "advance_time": self._handle_advance_time,
            "give_oxygen": self._handle_give_oxygen,
            "give_fluids": self._handle_give_fluids,
            "control_bleeding": self._handle_control_bleeding,
            "position_patient": self._handle_position_patient,
            "airway_support": self._handle_airway_support,
            "summarize_state": self._handle_summarize_state,
            "check_deterioration": self._handle_check_deterioration,
            "recommend_next_step": self._handle_recommend_next_step,
            "needle_decompression": self._handle_needle_decompression,
            "get_respiratory_status": self._handle_get_respiratory_status,
            "get_blood_gas": self._handle_get_blood_gas,
            "get_cbc": self._handle_get_cbc,
            "get_bmp": self._handle_get_bmp,
        }

    @property
    def available_tools(self) -> list[str]:
        return list(EXTENDED_TOOL_NAMES)

    def execute(self, action: PulsePhysiologyAction) -> ToolExecution:
        """Validate and execute a tool action."""

        tool_name = action.tool_name.strip()
        if tool_name not in self._handlers:
            return self._failure(
                tool_name=tool_name,
                state=self._adapter.get_full_state(),
                code="UNSUPPORTED_TOOL",
                message=f"Unsupported tool '{tool_name}'.",
                retryable=False,
            )

        try:
            return self._handlers[tool_name](dict(action.arguments))
        except ValueError as exc:
            return self._failure(
                tool_name=tool_name,
                state=self._adapter.get_full_state(),
                code="INVALID_ARGUMENT",
                message=str(exc),
                retryable=False,
            )
        except RuntimeError as exc:
            return self._failure(
                tool_name=tool_name,
                state=self._adapter.get_full_state(),
                code="ENGINE_ERROR",
                message=str(exc),
                retryable=True,
            )

    def _handle_get_vitals(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        message = (
            f"HR {self._fmt(state.heart_rate_bpm)} bpm, BP "
            f"{self._fmt(state.systolic_bp_mmhg)}/{self._fmt(state.diastolic_bp_mmhg)} mmHg, "
            f"SpO2 {self._fmt(state.spo2, precision=3)}, RR {self._fmt(state.respiration_rate_bpm)}."
        )
        return self._success("get_vitals", state, message, previous_state=state)

    def _handle_get_respiratory_status(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        message = (
            f"Breath sounds: {state.breath_sounds}; SpO2 {self._fmt(state.spo2, precision=3)}; "
            f"EtCO2 {self._fmt(state.etco2_mmhg)} mmHg; airway support: {state.airway_support or 'none'}."
        )
        return self._success("get_respiratory_status", state, message, previous_state=state)

    def _handle_get_blood_gas(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        abg = state.abg_result
        message = (
            f"ABG pH {self._fmt(abg.ph, precision=3)}, PaO2 {self._fmt(abg.partial_pressure_of_oxygen_mmhg)} mmHg, "
            f"PaCO2 {self._fmt(abg.partial_pressure_of_carbon_dioxide_mmhg)} mmHg, lactate "
            f"{self._fmt(abg.lactate_mg_per_dl)} mg/dL."
        )
        return self._success("get_blood_gas", state, message, previous_state=state)

    def _handle_get_cbc(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        cbc = state.cbc_result
        message = (
            f"CBC hemoglobin {self._fmt(cbc.hemoglobin_g_per_dl)} g/dL, hematocrit "
            f"{self._fmt(cbc.hematocrit_fraction, precision=3)}, WBC "
            f"{self._fmt(cbc.white_blood_cell_count_per_u_l)} /uL."
        )
        return self._success("get_cbc", state, message, previous_state=state)

    def _handle_get_bmp(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        bmp = state.bmp_result
        message = (
            f"BMP sodium {self._fmt(bmp.sodium_mmol_per_l)} mmol/L, potassium "
            f"{self._fmt(bmp.potassium_mmol_per_l)} mmol/L, creatinine "
            f"{self._fmt(bmp.creatinine_mg_per_dl)} mg/dL, glucose {self._fmt(bmp.glucose_mg_per_dl)} mg/dL."
        )
        return self._success("get_bmp", state, message, previous_state=state)

    def _handle_advance_time(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        seconds = self._read_positive_float(arguments, keys=("seconds", "duration_s", "sim_time_s"))
        if seconds <= 0:
            raise ValueError("seconds must be greater than zero.")
        if seconds > 1800:
            raise ValueError("seconds must be 1800 or less for a single tool call.")
        state = self._adapter.advance_time(seconds)
        return self._success(
            "advance_time",
            state,
            f"Advanced simulation by {seconds:.0f} seconds.",
            previous_state=previous,
        )

    def _handle_give_oxygen(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        device = str(arguments.get("device") or self._suggest_oxygen_device(previous))
        flow_lpm = self._read_optional_float(arguments, keys=("flow_lpm",))
        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0
        state = self._adapter.apply_supplemental_oxygen(
            device=device,
            flow_lpm=flow_lpm,
            advance_time_seconds=monitor_seconds,
        )
        resolved_flow = state.oxygen_flow_lpm if state.oxygen_flow_lpm is not None else flow_lpm
        return self._success(
            "give_oxygen",
            state,
            f"Supplemental oxygen applied via {state.oxygen_device or device} at {self._fmt(resolved_flow)} L/min.",
            previous_state=previous,
        )

    def _handle_give_fluids(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        fluid_type = str(arguments.get("fluid_type") or arguments.get("fluid") or "saline").strip().lower()
        compound_map = {
            "saline": "Saline",
            "blood": "Blood",
        }
        if fluid_type not in compound_map:
            valid = ", ".join(sorted(compound_map))
            raise ValueError(f"fluid_type must be one of: {valid}")

        volume_ml = self._read_optional_float(arguments, keys=("volume_ml", "bag_volume_ml")) or 500.0
        rate_ml_per_min = self._read_optional_float(arguments, keys=("rate_ml_per_min",)) or 100.0
        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",))
        state = self._adapter.infuse_compound(
            compound=compound_map[fluid_type],
            bag_volume_ml=volume_ml,
            rate_ml_per_min=rate_ml_per_min,
            advance_time_seconds=monitor_seconds,
        )
        return self._success(
            "give_fluids",
            state,
            f"Started {fluid_type} infusion: {volume_ml:.0f} mL at {rate_ml_per_min:.0f} mL/min.",
            previous_state=previous,
        )

    def _handle_control_bleeding(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        if not previous.active_hemorrhages:
            return self._failure(
                tool_name="control_bleeding",
                state=previous,
                code="PRECONDITION_FAILED",
                message="No active hemorrhage is currently tracked.",
                retryable=False,
            )

        site = str(arguments.get("site") or arguments.get("compartment") or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not site:
            if len(previous.active_hemorrhages) == 1:
                site = next(iter(previous.active_hemorrhages))
            else:
                return self._failure(
                    tool_name="control_bleeding",
                    state=previous,
                    code="INVALID_ARGUMENT",
                    message="Multiple active hemorrhages are present; specify a site.",
                    retryable=False,
                )

        if site not in previous.active_hemorrhages:
            return self._failure(
                tool_name="control_bleeding",
                state=previous,
                code="INVALID_ARGUMENT",
                message=f"'{site}' is not an active hemorrhage site.",
                retryable=False,
            )

        method = str(arguments.get("method") or self._default_bleeding_method(site)).strip().lower().replace("-", "_").replace(" ", "_")
        current_flow = previous.active_hemorrhages[site]
        if method == "tourniquet":
            new_flow = 0.0
        elif method == "pressure":
            new_flow = current_flow * 0.35
        elif method in {"hemostatic_dressing", "pressure_dressing"}:
            new_flow = current_flow * 0.2
        else:
            raise ValueError("method must be one of: tourniquet, pressure, hemostatic_dressing")

        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0
        state = self._adapter.set_hemorrhage(
            site,
            flow_rate_ml_per_min=new_flow,
            advance_time_seconds=monitor_seconds,
        )
        return self._success(
            "control_bleeding",
            state,
            f"Applied {method} to {site.replace('_', ' ')}.",
            previous_state=previous,
        )

    def _handle_position_patient(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        position = str(arguments.get("position") or self._suggest_position(previous))
        state = self._adapter.set_patient_position(position)
        return self._success(
            "position_patient",
            state,
            f"Patient position set to {state.position.replace('_', ' ')}.",
            previous_state=previous,
        )

    def _handle_airway_support(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        support_type = str(arguments.get("support_type") or arguments.get("mode") or self._suggest_airway_support(previous))
        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0
        state = self._adapter.set_intubation(support_type, advance_time_seconds=monitor_seconds)
        return self._success(
            "airway_support",
            state,
            f"Airway support set to {state.airway_support or 'off'}.",
            previous_state=previous,
        )

    def _handle_needle_decompression(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        side = str(arguments.get("side") or self._suggest_needle_side(previous))
        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",)) or 90.0
        state = self._adapter.apply_needle_decompression(side, advance_time_seconds=monitor_seconds)
        return self._success(
            "needle_decompression",
            state,
            f"Needle decompression performed on the {side.lower()} chest.",
            previous_state=previous,
        )

    def _handle_summarize_state(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        alerts = ", ".join(state.active_alerts) if state.active_alerts else "no active alerts"
        message = (
            f"{state.scenario_id}: HR {self._fmt(state.heart_rate_bpm)}, MAP {self._fmt(state.mean_arterial_pressure_mmhg)}, "
            f"SpO2 {self._fmt(state.spo2, precision=3)}, mental status {state.mental_status}, alerts {alerts}."
        )
        return self._success("summarize_state", state, message, previous_state=state)

    def _handle_check_deterioration(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        deterioration_domains: list[str] = []
        if state.spo2 is not None and state.spo2 < 0.92:
            deterioration_domains.append("respiratory")
        if state.shock_index is not None and state.shock_index >= 0.9:
            deterioration_domains.append("circulatory")
        if state.mental_status in {"pain", "unresponsive"}:
            deterioration_domains.append("neurologic")
        if state.lactate_trend == "worsening":
            deterioration_domains.append("perfusion")
        if not deterioration_domains:
            message = "No major deterioration flags detected right now."
        else:
            message = f"Deterioration detected in: {', '.join(dict.fromkeys(deterioration_domains))}."
        return self._success("check_deterioration", state, message, previous_state=state)

    def _handle_recommend_next_step(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        recommendation = self._recommend_next_tool(state)
        return self._success(
            "recommend_next_step",
            state,
            f"Recommended next step: {recommendation}.",
            previous_state=state,
        )

    def _recommend_next_tool(self, state: PatientState) -> str:
        if "possible_tension_pneumothorax" in state.active_alerts or "unilateral_absent_breath_sounds" in state.active_alerts:
            return "needle_decompression"
        if state.active_hemorrhages:
            return "control_bleeding"
        if state.spo2 is not None and state.spo2 < 0.92:
            return "give_oxygen"
        if state.mean_arterial_pressure_mmhg is not None and state.mean_arterial_pressure_mmhg < 65:
            return "give_fluids"
        if state.mental_status in {"pain", "unresponsive"} and not state.intubated:
            return "airway_support"
        return "get_vitals"

    @staticmethod
    def _suggest_oxygen_device(state: PatientState) -> str:
        if state.spo2 is not None and state.spo2 < 0.9:
            return "non_rebreather_mask"
        if state.spo2 is not None and state.spo2 < 0.95:
            return "simple_mask"
        return "nasal_cannula"

    @staticmethod
    def _default_bleeding_method(site: str) -> str:
        if site in {"left_arm", "right_arm", "left_leg", "right_leg"}:
            return "tourniquet"
        return "pressure"

    @staticmethod
    def _suggest_position(state: PatientState) -> str:
        if state.spo2 is not None and state.spo2 < 0.92:
            return "upright"
        return "supine"

    @staticmethod
    def _suggest_airway_support(state: PatientState) -> str:
        if state.mental_status in {"pain", "unresponsive"}:
            return "tracheal"
        if state.mental_status == "verbal":
            return "oropharyngeal"
        return "nasopharyngeal"

    @staticmethod
    def _suggest_needle_side(state: PatientState) -> str:
        if "absent left" in state.breath_sounds:
            return "left"
        if "absent right" in state.breath_sounds:
            return "right"
        return "left"

    @staticmethod
    def _read_positive_float(arguments: dict[str, Any], *, keys: tuple[str, ...]) -> float:
        value = PulseToolExecutor._read_optional_float(arguments, keys=keys)
        if value is None:
            joined = ", ".join(keys)
            raise ValueError(f"One of {joined} is required.")
        return value

    @staticmethod
    def _read_optional_float(arguments: dict[str, Any], *, keys: tuple[str, ...]) -> float | None:
        for key in keys:
            if key in arguments and arguments[key] is not None:
                return float(arguments[key])
        return None

    @staticmethod
    def _fmt(value: float | None, *, precision: int = 1) -> str:
        if value is None:
            return "n/a"
        return f"{value:.{precision}f}"

    @staticmethod
    def _changed_fields(previous_state: PatientState, current_state: PatientState) -> list[str]:
        previous_dump = previous_state.model_dump()
        current_dump = current_state.model_dump()
        return [
            field_name
            for field_name in current_dump
            if previous_dump.get(field_name) != current_dump.get(field_name)
        ]

    def _success(
        self,
        tool_name: str,
        state: PatientState,
        message: str,
        *,
        previous_state: PatientState,
    ) -> ToolExecution:
        changed_fields = self._changed_fields(previous_state, state)
        return ToolExecution(
            state=state,
            tool_result=ToolResult(
                tool_name=tool_name,
                success=True,
                message=message,
                state_changed=bool(changed_fields),
                changed_fields=changed_fields,
            ),
            error=None,
        )

    def _failure(
        self,
        *,
        tool_name: str,
        state: PatientState,
        code: str,
        message: str,
        retryable: bool,
    ) -> ToolExecution:
        error = ToolError(code=code, message=message, retryable=retryable)
        return ToolExecution(
            state=state,
            tool_result=ToolResult(
                tool_name=tool_name,
                success=False,
                message=message,
                state_changed=False,
                changed_fields=[],
            ),
            error=error,
        )
