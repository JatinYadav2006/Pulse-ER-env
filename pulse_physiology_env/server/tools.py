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
    "give_pressor",
    "needle_decompression",
    "pericardiocentesis",
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
            "give_pressor": self._handle_give_pressor,
            "needle_decompression": self._handle_needle_decompression,
            "pericardiocentesis": self._handle_pericardiocentesis,
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
        return self._handle_delayed_diagnostic(
            tool_name="get_blood_gas",
            label="ABG",
            render_result=lambda state: (
                f"ABG pH {self._fmt(state.abg_result.ph, precision=3)}, PaO2 "
                f"{self._fmt(state.abg_result.partial_pressure_of_oxygen_mmhg)} mmHg, PaCO2 "
                f"{self._fmt(state.abg_result.partial_pressure_of_carbon_dioxide_mmhg)} mmHg, lactate "
                f"{self._fmt(state.abg_result.lactate_mg_per_dl)} mg/dL."
            ),
        )

    def _handle_get_cbc(self, _arguments: dict[str, Any]) -> ToolExecution:
        return self._handle_delayed_diagnostic(
            tool_name="get_cbc",
            label="CBC",
            render_result=lambda state: (
                f"CBC hemoglobin {self._fmt(state.cbc_result.hemoglobin_g_per_dl)} g/dL, hematocrit "
                f"{self._fmt(state.cbc_result.hematocrit_fraction, precision=3)}, WBC "
                f"{self._fmt(state.cbc_result.white_blood_cell_count_per_u_l)} /uL."
            ),
        )

    def _handle_get_bmp(self, _arguments: dict[str, Any]) -> ToolExecution:
        return self._handle_delayed_diagnostic(
            tool_name="get_bmp",
            label="BMP",
            render_result=lambda state: (
                f"BMP sodium {self._fmt(state.bmp_result.sodium_mmol_per_l)} mmol/L, potassium "
                f"{self._fmt(state.bmp_result.potassium_mmol_per_l)} mmol/L, creatinine "
                f"{self._fmt(state.bmp_result.creatinine_mg_per_dl)} mg/dL, glucose "
                f"{self._fmt(state.bmp_result.glucose_mg_per_dl)} mg/dL."
            ),
        )

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
        volume_ml = self._read_optional_float(arguments, keys=("volume_ml", "bag_volume_ml")) or 500.0
        rate_ml_per_min = self._read_optional_float(arguments, keys=("rate_ml_per_min",)) or 100.0
        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0
        compound = self._adapter.resolve_fluid_compound(fluid_type)
        state = self._adapter.infuse_compound(
            compound=compound,
            bag_volume_ml=volume_ml,
            rate_ml_per_min=rate_ml_per_min,
            advance_time_seconds=monitor_seconds,
        )
        return self._success(
            "give_fluids",
            state,
            f"Started {compound} infusion: {volume_ml:.0f} mL at {rate_ml_per_min:.0f} mL/min.",
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
        support_type = str(
            arguments.get("support_type") or arguments.get("mode") or self._suggest_airway_support(previous)
        )
        support_key = support_type.strip().lower().replace("-", "_").replace(" ", "_")
        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0
        if support_key in {"bag_valve_mask", "bvm"}:
            state = self._adapter.apply_bag_valve_mask(
                fio2=self._read_optional_float(arguments, keys=("fio2", "fraction_inspired_oxygen")),
                peep_cmh2o=self._read_optional_float(arguments, keys=("peep_cmh2o", "peep")),
                respiration_rate_bpm=self._read_optional_float(arguments, keys=("respiration_rate_bpm", "rate_bpm")),
                inspiratory_expiratory_ratio=self._read_optional_float(arguments, keys=("ie_ratio", "inspiratory_expiratory_ratio")),
                squeeze_pressure_cmh2o=self._read_optional_float(arguments, keys=("squeeze_pressure_cmh2o", "pressure_cmh2o")),
                squeeze_volume_ml=self._read_optional_float(arguments, keys=("squeeze_volume_ml", "tidal_volume_ml")),
                airway_adjunct=arguments.get("airway_adjunct"),
                advance_time_seconds=monitor_seconds,
            )
        elif support_key == "cpap":
            state = self._adapter.apply_cpap(
                fio2=self._read_optional_float(arguments, keys=("fio2", "fraction_inspired_oxygen")),
                peep_cmh2o=self._read_optional_float(arguments, keys=("peep_cmh2o", "peep")),
                pressure_support_cmh2o=self._read_optional_float(arguments, keys=("pressure_support_cmh2o", "pressure_support")),
                advance_time_seconds=monitor_seconds,
            )
        elif support_key in {"pressure_control_ventilation", "pressure_control", "ventilator", "mechanical_ventilation"}:
            state = self._adapter.apply_pressure_control_ventilation(
                fio2=self._read_optional_float(arguments, keys=("fio2", "fraction_inspired_oxygen")),
                peep_cmh2o=self._read_optional_float(arguments, keys=("peep_cmh2o", "peep")),
                inspiratory_pressure_cmh2o=self._read_optional_float(arguments, keys=("inspiratory_pressure_cmh2o", "pressure_cmh2o")),
                respiration_rate_bpm=self._read_optional_float(arguments, keys=("respiration_rate_bpm", "rate_bpm")),
                inspiratory_period_s=self._read_optional_float(arguments, keys=("inspiratory_period_s",)),
                advance_time_seconds=monitor_seconds,
            )
        else:
            state = self._adapter.set_intubation(support_type, advance_time_seconds=monitor_seconds)
        return self._success(
            "airway_support",
            state,
            f"Airway support set to {state.airway_support or 'off'}.",
            previous_state=previous,
        )

    def _handle_give_pressor(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        pressor = str(arguments.get("pressor") or arguments.get("agent") or "norepinephrine")
        rate_ml_per_min = self._read_optional_float(arguments, keys=("rate_ml_per_min",))
        concentration_ug_per_ml = self._read_optional_float(arguments, keys=("concentration_ug_per_ml",))
        if bool(arguments.get("stop")):
            rate_ml_per_min = 0.0
        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0
        state = self._adapter.set_pressor(
            pressor=pressor,
            concentration_ug_per_ml=concentration_ug_per_ml,
            rate_ml_per_min=rate_ml_per_min,
            advance_time_seconds=monitor_seconds,
        )
        if rate_ml_per_min == 0.0:
            message = f"Stopped {pressor} infusion."
        else:
            pressor_key = pressor.strip().lower().replace("-", "_").replace(" ", "_")
            active_rate = state.active_infusions.get(pressor_key)
            rate_text = self._fmt(active_rate) if active_rate is not None else self._fmt(rate_ml_per_min)
            concentration_text = (
                self._fmt(concentration_ug_per_ml)
                if concentration_ug_per_ml is not None
                else "default"
            )
            message = (
                f"Started {pressor} at {rate_text} mL/min with concentration "
                f"{concentration_text} ug/mL."
            )
        return self._success("give_pressor", state, message, previous_state=previous)

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

    def _handle_pericardiocentesis(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        drain_rate_ml_per_min = (
            self._read_optional_float(arguments, keys=("drain_rate_ml_per_min", "rate_ml_per_min")) or 150.0
        )
        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",)) or 180.0
        state = self._adapter.perform_pericardiocentesis(
            drain_rate_ml_per_min=drain_rate_ml_per_min,
            advance_time_seconds=monitor_seconds,
        )
        return self._success(
            "pericardiocentesis",
            state,
            f"Pericardiocentesis performed with drainage at {self._fmt(drain_rate_ml_per_min)} mL/min.",
            previous_state=previous,
        )

    def _handle_summarize_state(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        alerts = ", ".join(state.active_alerts) if state.active_alerts else "no active alerts"
        pending = ", ".join(
            f"{name}:{seconds}s" for name, seconds in sorted(state.pending_diagnostics.items())
        ) or "none"
        ready = ", ".join(state.ready_diagnostics) if state.ready_diagnostics else "none"
        message = (
            f"{state.scenario_id} ({state.scenario_difficulty}): HR {self._fmt(state.heart_rate_bpm)}, "
            f"MAP {self._fmt(state.mean_arterial_pressure_mmhg)}, SpO2 {self._fmt(state.spo2, precision=3)}, "
            f"mental status {state.mental_status}, alerts {alerts}, pending diagnostics {pending}, ready diagnostics {ready}."
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
        if state.pending_diagnostics:
            deterioration_domains.append("diagnostics")
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
        if "possible_cardiac_tamponade" in state.active_alerts:
            return "pericardiocentesis"
        if state.active_hemorrhages:
            return "control_bleeding"
        if state.spo2 is not None and state.spo2 < 0.88 and state.airway_support not in {
            "bag_valve_mask",
            "pressure_control_ventilation",
        }:
            return "airway_support"
        if state.spo2 is not None and state.spo2 < 0.92:
            return "give_oxygen"
        if state.mean_arterial_pressure_mmhg is not None and state.mean_arterial_pressure_mmhg < 65:
            if "blood" in state.active_infusions or "saline" in state.active_infusions or "packed_rbc" in state.active_infusions:
                return "give_pressor"
            return "give_fluids"
        if state.ready_diagnostics and "get_blood_gas" in state.ready_diagnostics:
            return "get_blood_gas"
        if state.mental_status in {"pain", "unresponsive"} and state.airway_support is None:
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
        if state.spo2 is not None and state.spo2 < 0.85:
            return "pressure_control_ventilation" if state.mental_status in {"pain", "unresponsive"} else "bag_valve_mask"
        if state.spo2 is not None and state.spo2 < 0.9:
            return "cpap" if state.mental_status in {"alert", "verbal"} else "bag_valve_mask"
        if state.mental_status in {"pain", "unresponsive"}:
            return "tracheal"
        if state.mental_status == "verbal":
            return "oropharyngeal"
        return "nasopharyngeal"

    def _handle_delayed_diagnostic(
        self,
        *,
        tool_name: str,
        label: str,
        render_result: Callable[[PatientState], str],
    ) -> ToolExecution:
        previous = self._adapter.get_full_state()
        if tool_name in previous.ready_diagnostics:
            return self._success(tool_name, previous, render_result(previous), previous_state=previous)

        if tool_name in previous.pending_diagnostics:
            remaining = previous.pending_diagnostics[tool_name]
            return self._success(
                tool_name,
                previous,
                f"{label} is pending. {remaining} simulated seconds remaining before results are ready.",
                previous_state=previous,
            )

        state = self._adapter.order_diagnostic(tool_name)
        remaining = state.pending_diagnostics.get(tool_name)
        return self._success(
            tool_name,
            state,
            f"Ordered {label}. Results will be ready after about {remaining} simulated seconds.",
            previous_state=previous,
        )

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
