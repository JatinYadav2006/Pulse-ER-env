"""Tool registry and execution logic for the Pulse physiology environment."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from pulse_physiology_env.models import PulsePhysiologyAction, ToolError, ToolResult
from pulse_physiology_env.patient_state import PatientState

from .pulse_engine_adapter import PulseEngineAdapter


def _unique_tool_names(*groups: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for name in group:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
    return ordered


SPEC_PRIORITY_TOOL_NAMES = [
    "get_vitals",
    "get_hemodynamics",
    "get_blood_chemistry",
    "get_respiratory_state",
    "get_shock_assessment",
    "perform_needle_decompression",
    "administer_crystalloid_bolus",
    "start_norepinephrine_infusion",
    "administer_epinephrine_bolus",
    "perform_intubation",
    "run_triage_assessment",
    "detect_deterioration",
]

CLINICAL_45_TOOL_NAMES = [
    "apply_nasal_cannula",
    "apply_simple_mask",
    "apply_nonrebreather_mask",
    "apply_bag_valve_mask",
    "perform_intubation",
    "set_ventilator_tidal_volume",
    "set_ventilator_rate",
    "set_ventilator_fio2",
    "initiate_hemorrhage",
    "apply_tourniquet",
    "apply_wound_packing",
    "apply_direct_pressure",
    "perform_needle_decompression",
    "perform_pericardiocentesis",
    "perform_cpr",
    "induce_cardiac_arrest",
    "apply_pericardial_effusion",
    "administer_epinephrine_bolus",
    "administer_morphine_bolus",
    "administer_ketamine_bolus",
    "administer_midazolam_bolus",
    "administer_lorazepam_bolus",
    "administer_succinylcholine_bolus",
    "administer_phenylephrine_bolus",
    "administer_atropine_bolus",
    "start_norepinephrine_infusion",
    "start_dopamine_infusion",
    "start_phenylephrine_infusion",
    "adjust_infusion_rate",
    "stop_infusion",
    "administer_crystalloid_bolus",
    "administer_blood_transfusion",
    "administer_plasma",
    "activate_massive_transfusion_protocol",
    "auscultate_chest",
    "assess_consciousness_level",
    "check_pain_level",
    "measure_core_temperature",
    "check_end_tidal_co2",
    "calculate_shock_index",
    "order_arterial_blood_gas",
    "order_complete_blood_count",
    "order_basic_metabolic_panel",
    "order_point_of_care_ultrasound",
    "assess_urine_output",
]

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

LEGACY_EXTENDED_TOOL_NAMES = INITIAL_TOOL_NAMES + [
    "give_pressor",
    "needle_decompression",
    "pericardiocentesis",
    "get_respiratory_status",
    "get_blood_gas",
    "get_cbc",
    "get_bmp",
]

AVAILABLE_TOOL_NAMES = _unique_tool_names(
    CLINICAL_45_TOOL_NAMES,
    SPEC_PRIORITY_TOOL_NAMES,
    LEGACY_EXTENDED_TOOL_NAMES,
)

RESTRICTED_TOOL_NAMES = {
    "initiate_hemorrhage",
    "induce_cardiac_arrest",
    "apply_pericardial_effusion",
}


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
            "get_hemodynamics": self._handle_get_hemodynamics,
            "get_blood_chemistry": self._handle_get_blood_chemistry,
            "get_respiratory_state": partial(self._handle_get_respiratory_status, tool_name="get_respiratory_state"),
            "get_respiratory_status": partial(self._handle_get_respiratory_status, tool_name="get_respiratory_status"),
            "get_shock_assessment": self._handle_get_shock_assessment,
            "run_triage_assessment": self._handle_run_triage_assessment,
            "detect_deterioration": partial(self._handle_check_deterioration, tool_name="detect_deterioration"),
            "check_deterioration": partial(self._handle_check_deterioration, tool_name="check_deterioration"),
            "advance_time": self._handle_advance_time,
            "give_oxygen": self._handle_give_oxygen,
            "apply_nasal_cannula": partial(self._handle_fixed_oxygen, tool_name="apply_nasal_cannula", device="nasal_cannula"),
            "apply_simple_mask": partial(self._handle_fixed_oxygen, tool_name="apply_simple_mask", device="simple_mask"),
            "apply_nonrebreather_mask": partial(self._handle_fixed_oxygen, tool_name="apply_nonrebreather_mask", device="non_rebreather_mask"),
            "give_fluids": self._handle_give_fluids,
            "administer_crystalloid_bolus": self._handle_administer_crystalloid_bolus,
            "administer_blood_transfusion": self._handle_administer_blood_transfusion,
            "administer_plasma": self._handle_administer_plasma,
            "activate_massive_transfusion_protocol": self._handle_activate_massive_transfusion_protocol,
            "control_bleeding": self._handle_control_bleeding,
            "apply_tourniquet": self._handle_apply_tourniquet,
            "apply_wound_packing": self._handle_apply_wound_packing,
            "apply_direct_pressure": self._handle_apply_direct_pressure,
            "initiate_hemorrhage": self._handle_initiate_hemorrhage,
            "position_patient": self._handle_position_patient,
            "airway_support": self._handle_airway_support,
            "apply_bag_valve_mask": self._handle_apply_bag_valve_mask,
            "perform_intubation": self._handle_perform_intubation,
            "set_ventilator_tidal_volume": self._handle_set_ventilator_tidal_volume,
            "set_ventilator_rate": self._handle_set_ventilator_rate,
            "set_ventilator_fio2": self._handle_set_ventilator_fio2,
            "give_pressor": self._handle_give_pressor,
            "start_norepinephrine_infusion": partial(self._handle_start_pressor_infusion, tool_name="start_norepinephrine_infusion", pressor="norepinephrine"),
            "start_dopamine_infusion": partial(self._handle_start_pressor_infusion, tool_name="start_dopamine_infusion", pressor="dopamine"),
            "start_phenylephrine_infusion": partial(self._handle_start_pressor_infusion, tool_name="start_phenylephrine_infusion", pressor="phenylephrine"),
            "adjust_infusion_rate": self._handle_adjust_infusion_rate,
            "stop_infusion": self._handle_stop_infusion,
            "needle_decompression": self._handle_needle_decompression,
            "perform_needle_decompression": self._handle_perform_needle_decompression,
            "pericardiocentesis": self._handle_pericardiocentesis,
            "perform_pericardiocentesis": self._handle_perform_pericardiocentesis,
            "perform_cpr": self._handle_perform_cpr,
            "induce_cardiac_arrest": self._handle_induce_cardiac_arrest,
            "apply_pericardial_effusion": self._handle_apply_pericardial_effusion,
            "administer_epinephrine_bolus": partial(self._handle_bolus_drug, tool_name="administer_epinephrine_bolus", drug_key="epinephrine", dose_keys=("dose_mg", "dose")),
            "administer_morphine_bolus": partial(self._handle_bolus_drug, tool_name="administer_morphine_bolus", drug_key="morphine", dose_keys=("dose_mg", "dose")),
            "administer_ketamine_bolus": partial(self._handle_bolus_drug, tool_name="administer_ketamine_bolus", drug_key="ketamine", dose_keys=("dose_mg_per_kg", "dose")),
            "administer_midazolam_bolus": partial(self._handle_bolus_drug, tool_name="administer_midazolam_bolus", drug_key="midazolam", dose_keys=("dose_mg", "dose")),
            "administer_lorazepam_bolus": partial(self._handle_bolus_drug, tool_name="administer_lorazepam_bolus", drug_key="lorazepam", dose_keys=("dose_mg", "dose")),
            "administer_succinylcholine_bolus": partial(self._handle_bolus_drug, tool_name="administer_succinylcholine_bolus", drug_key="succinylcholine", dose_keys=("dose_mg_per_kg", "dose")),
            "administer_phenylephrine_bolus": partial(self._handle_bolus_drug, tool_name="administer_phenylephrine_bolus", drug_key="phenylephrine", dose_keys=("dose_mcg", "dose")),
            "administer_atropine_bolus": partial(self._handle_bolus_drug, tool_name="administer_atropine_bolus", drug_key="atropine", dose_keys=("dose_mg", "dose")),
            "summarize_state": self._handle_summarize_state,
            "recommend_next_step": self._handle_recommend_next_step,
            "get_blood_gas": partial(self._handle_get_blood_gas, tool_name="get_blood_gas"),
            "get_cbc": partial(self._handle_get_cbc, tool_name="get_cbc"),
            "get_bmp": partial(self._handle_get_bmp, tool_name="get_bmp"),
            "order_arterial_blood_gas": partial(self._handle_get_blood_gas, tool_name="order_arterial_blood_gas"),
            "order_complete_blood_count": partial(self._handle_get_cbc, tool_name="order_complete_blood_count"),
            "order_basic_metabolic_panel": partial(self._handle_get_bmp, tool_name="order_basic_metabolic_panel"),
            "order_point_of_care_ultrasound": self._handle_order_point_of_care_ultrasound,
            "auscultate_chest": self._handle_auscultate_chest,
            "assess_consciousness_level": self._handle_assess_consciousness_level,
            "check_pain_level": self._handle_check_pain_level,
            "measure_core_temperature": self._handle_measure_core_temperature,
            "check_end_tidal_co2": self._handle_check_end_tidal_co2,
            "calculate_shock_index": self._handle_calculate_shock_index,
            "assess_urine_output": self._handle_assess_urine_output,
        }

    @property
    def available_tools(self) -> list[str]:
        return list(AVAILABLE_TOOL_NAMES)

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
            f"MAP {self._fmt(state.mean_arterial_pressure_mmhg)} mmHg, SpO2 {self._fmt(state.spo2, precision=3)}, "
            f"RR {self._fmt(state.respiration_rate_bpm)}, Temp {self._fmt(state.core_temperature_c)} C, "
            f"EtCO2 {self._fmt(state.etco2_mmhg)} mmHg."
        )
        return self._success("get_vitals", state, message, previous_state=state)

    def _handle_get_hemodynamics(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        hemodynamics = self._adapter.get_hemodynamics_summary()
        message = (
            f"CO {self._fmt(hemodynamics['cardiac_output_l_per_min'])} L/min, stroke volume "
            f"{self._fmt(hemodynamics['stroke_volume_ml'])} mL, SVR "
            f"{self._fmt(hemodynamics['systemic_vascular_resistance_mmhg_min_per_l'])} mmHg*min/L, "
            f"MAP {self._fmt(hemodynamics['mean_arterial_pressure_mmhg'])} mmHg."
        )
        return self._success("get_hemodynamics", state, message, previous_state=state)

    def _handle_get_blood_chemistry(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        chemistry = self._adapter.get_blood_chemistry_summary()
        message = (
            f"pH {self._fmt(chemistry['ph'], precision=3)}, lactate {self._fmt(chemistry['lactate_mg_per_dl'])} mg/dL, "
            f"hemoglobin {self._fmt(chemistry['hemoglobin_g_per_dl'])} g/dL, base deficit "
            f"{self._fmt(chemistry['base_deficit_meq_per_l'])} mEq/L."
        )
        return self._success("get_blood_chemistry", state, message, previous_state=state)

    def _handle_get_respiratory_status(self, _arguments: dict[str, Any], *, tool_name: str) -> ToolExecution:
        state = self._adapter.get_full_state()
        message = (
            f"Breath sounds: {state.breath_sounds}; SpO2 {self._fmt(state.spo2, precision=3)}; "
            f"EtCO2 {self._fmt(state.etco2_mmhg)} mmHg; tidal volume {self._fmt(state.tidal_volume_ml)} mL; "
            f"airway support: {state.airway_support or 'none'}."
        )
        return self._success(tool_name, state, message, previous_state=state)

    def _handle_get_shock_assessment(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        shock = self._adapter.get_shock_assessment()
        message = (
            f"Shock index {self._fmt(shock['shock_index'], precision=2)}, class {shock['shock_class']}, "
            f"perfusion {shock['perfusion_status']}, urine output {self._fmt(shock['urine_output_ml_per_hr'])} mL/hr."
        )
        return self._success("get_shock_assessment", state, message, previous_state=state)

    def _handle_run_triage_assessment(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        shock = self._adapter.get_shock_assessment()
        respiratory = "unstable" if state.spo2 is not None and state.spo2 < 0.9 else "acceptable"
        if "cardiac_arrest" in state.active_alerts:
            triage = "expectant"
        elif state.mean_arterial_pressure_mmhg is not None and state.mean_arterial_pressure_mmhg < 55:
            triage = "immediate"
        elif state.spo2 is not None and state.spo2 < 0.9:
            triage = "immediate"
        elif state.active_hemorrhages:
            triage = "urgent"
        else:
            triage = "delayed"
        alerts = ", ".join(state.active_alerts) if state.active_alerts else "no active alerts"
        message = (
            f"Triage {triage}. Respiratory status {respiratory}; shock class {shock['shock_class']}; "
            f"mental status {state.mental_status}; alerts {alerts}."
        )
        return self._success("run_triage_assessment", state, message, previous_state=state)

    def _handle_check_deterioration(self, _arguments: dict[str, Any], *, tool_name: str) -> ToolExecution:
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
        return self._success(tool_name, state, message, previous_state=state)

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

    def _handle_fixed_oxygen(self, arguments: dict[str, Any], *, tool_name: str, device: str) -> ToolExecution:
        previous = self._adapter.get_full_state()
        flow_lpm = self._read_optional_float(arguments, keys=("flow_rate_lpm", "flow_lpm")) or self._default_oxygen_flow(device)
        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0
        state = self._adapter.apply_supplemental_oxygen(
            device=device,
            flow_lpm=flow_lpm,
            advance_time_seconds=monitor_seconds,
        )
        return self._success(
            tool_name,
            state,
            f"Applied {device.replace('_', ' ')} at {self._fmt(flow_lpm)} L/min.",
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

    def _handle_administer_crystalloid_bolus(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        volume_l = self._read_optional_float(arguments, keys=("volume_l",))
        volume_ml = self._read_optional_float(arguments, keys=("volume_ml", "bag_volume_ml"))
        resolved_volume_ml = volume_ml if volume_ml is not None else (volume_l * 1000.0 if volume_l is not None else 1000.0)
        rate_ml_per_min = self._read_optional_float(arguments, keys=("rate_ml_per_min",)) or 250.0
        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",)) or 120.0
        state = self._adapter.infuse_compound(
            compound="Saline",
            bag_volume_ml=resolved_volume_ml,
            rate_ml_per_min=rate_ml_per_min,
            advance_time_seconds=monitor_seconds,
        )
        return self._success(
            "administer_crystalloid_bolus",
            state,
            f"Administered crystalloid bolus of {resolved_volume_ml:.0f} mL at {rate_ml_per_min:.0f} mL/min.",
            previous_state=previous,
        )

    def _handle_administer_blood_transfusion(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        units = self._read_optional_float(arguments, keys=("units",)) or 1.0
        if units <= 0:
            raise ValueError("units must be greater than zero.")
        bag_volume_ml = units * 300.0
        rate_ml_per_min = self._read_optional_float(arguments, keys=("rate_ml_per_min",)) or 150.0
        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",)) or 120.0
        state = self._adapter.infuse_compound(
            compound="PackedRBC",
            bag_volume_ml=bag_volume_ml,
            rate_ml_per_min=rate_ml_per_min,
            advance_time_seconds=monitor_seconds,
        )
        return self._success(
            "administer_blood_transfusion",
            state,
            f"Started packed red blood cell transfusion for {units:.1f} unit(s).",
            previous_state=previous,
        )

    def _handle_administer_plasma(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        if not self._adapter.supports_compound("Plasma"):
            return self._failure(
                tool_name="administer_plasma",
                state=previous,
                code="UNSUPPORTED_BY_ENGINE",
                message="The local Pulse build does not expose a plasma compound infusion.",
                retryable=False,
            )
        units = self._read_optional_float(arguments, keys=("units",)) or 1.0
        if units <= 0:
            raise ValueError("units must be greater than zero.")
        bag_volume_ml = units * 250.0
        rate_ml_per_min = self._read_optional_float(arguments, keys=("rate_ml_per_min",)) or 125.0
        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",)) or 120.0
        state = self._adapter.infuse_compound(
            compound="Plasma",
            bag_volume_ml=bag_volume_ml,
            rate_ml_per_min=rate_ml_per_min,
            advance_time_seconds=monitor_seconds,
        )
        return self._success(
            "administer_plasma",
            state,
            f"Started plasma transfusion for {units:.1f} unit(s).",
            previous_state=previous,
        )

    def _handle_activate_massive_transfusion_protocol(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        if not self._adapter.supports_compound("Plasma"):
            return self._failure(
                tool_name="activate_massive_transfusion_protocol",
                state=previous,
                code="UNSUPPORTED_BY_ENGINE",
                message="Full MTP is unavailable because the local Pulse build does not expose plasma.",
                retryable=False,
            )
        prbc_units = self._read_optional_float(arguments, keys=("prbc_units",)) or 2.0
        plasma_units = self._read_optional_float(arguments, keys=("plasma_units",)) or 2.0
        rate_ml_per_min = self._read_optional_float(arguments, keys=("rate_ml_per_min",)) or 200.0
        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",)) or 180.0
        self._adapter.infuse_compound(
            compound="PackedRBC",
            bag_volume_ml=prbc_units * 300.0,
            rate_ml_per_min=rate_ml_per_min,
            advance_time_seconds=0.0,
        )
        state = self._adapter.infuse_compound(
            compound="Plasma",
            bag_volume_ml=plasma_units * 250.0,
            rate_ml_per_min=rate_ml_per_min,
            advance_time_seconds=monitor_seconds,
        )
        return self._success(
            "activate_massive_transfusion_protocol",
            state,
            f"Activated MTP with {prbc_units:.1f} unit(s) PRBC and {plasma_units:.1f} unit(s) plasma.",
            previous_state=previous,
        )

    def _handle_control_bleeding(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        active_site = self._resolve_bleeding_site(previous, arguments)
        if active_site is None:
            return self._failure(
                tool_name="control_bleeding",
                state=previous,
                code="PRECONDITION_FAILED",
                message="No active hemorrhage is currently tracked.",
                retryable=False,
            )
        method = str(arguments.get("method") or self._default_bleeding_method(active_site)).strip().lower().replace("-", "_").replace(" ", "_")
        current_flow = previous.active_hemorrhages[active_site]
        if method == "tourniquet":
            new_flow = 0.0
        elif method == "pressure":
            new_flow = current_flow * 0.35
        elif method in {"hemostatic_dressing", "pressure_dressing", "wound_packing"}:
            new_flow = current_flow * 0.2
        else:
            raise ValueError("method must be one of: tourniquet, pressure, hemostatic_dressing")
        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0
        state = self._adapter.set_hemorrhage(
            active_site,
            flow_rate_ml_per_min=new_flow,
            advance_time_seconds=monitor_seconds,
        )
        return self._success(
            "control_bleeding",
            state,
            f"Applied {method} to {active_site.replace('_', ' ')}.",
            previous_state=previous,
        )

    def _handle_apply_tourniquet(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        limb = self._normalize_site(arguments.get("limb") or arguments.get("site"))
        if not limb:
            raise ValueError("limb is required and must be one of: left_arm, right_arm, left_leg, right_leg")
        if limb not in {"left_arm", "right_arm", "left_leg", "right_leg"}:
            raise ValueError("limb must be one of: left_arm, right_arm, left_leg, right_leg")
        if limb not in previous.active_hemorrhages:
            return self._failure(
                tool_name="apply_tourniquet",
                state=previous,
                code="PRECONDITION_FAILED",
                message=f"No active limb hemorrhage is tracked at '{limb}'.",
                retryable=False,
            )
        state = self._adapter.set_hemorrhage(limb, flow_rate_ml_per_min=0.0, advance_time_seconds=self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0)
        return self._success("apply_tourniquet", state, f"Tourniquet applied to {limb.replace('_', ' ')}.", previous_state=previous)

    def _handle_apply_wound_packing(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        site = self._resolve_bleeding_site(previous, arguments)
        if site is None:
            return self._failure(
                tool_name="apply_wound_packing",
                state=previous,
                code="PRECONDITION_FAILED",
                message="No active hemorrhage is currently tracked.",
                retryable=False,
            )
        state = self._adapter.set_hemorrhage(
            site,
            flow_rate_ml_per_min=previous.active_hemorrhages[site] * 0.2,
            advance_time_seconds=self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0,
        )
        return self._success("apply_wound_packing", state, f"Wound packing applied to {site.replace('_', ' ')}.", previous_state=previous)

    def _handle_apply_direct_pressure(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        site = self._resolve_bleeding_site(previous, arguments)
        if site is None:
            return self._failure(
                tool_name="apply_direct_pressure",
                state=previous,
                code="PRECONDITION_FAILED",
                message="No active hemorrhage is currently tracked.",
                retryable=False,
            )
        state = self._adapter.set_hemorrhage(
            site,
            flow_rate_ml_per_min=previous.active_hemorrhages[site] * 0.35,
            advance_time_seconds=self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0,
        )
        return self._success("apply_direct_pressure", state, f"Direct pressure applied to {site.replace('_', ' ')}.", previous_state=previous)

    def _handle_initiate_hemorrhage(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        restriction = self._restricted_tool_failure("initiate_hemorrhage", previous, arguments)
        if restriction is not None:
            return restriction
        compartment = str(arguments.get("compartment") or "").strip()
        rate_ml_per_min = self._read_positive_float(arguments, keys=("rate_ml_per_min", "rate"))
        hemorrhage_type = str(arguments.get("hemorrhage_type") or "external")
        state = self._adapter.set_hemorrhage(
            compartment,
            flow_rate_ml_per_min=rate_ml_per_min,
            hemorrhage_type=hemorrhage_type,
            advance_time_seconds=self._read_optional_float(arguments, keys=("monitor_seconds",)) or 30.0,
        )
        return self._success(
            "initiate_hemorrhage",
            state,
            f"Initiated {hemorrhage_type} hemorrhage in {compartment} at {rate_ml_per_min:.0f} mL/min.",
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
        elif support_key in {"pressure_control_ventilation", "pressure_control"}:
            state = self._adapter.apply_pressure_control_ventilation(
                fio2=self._read_optional_float(arguments, keys=("fio2", "fraction_inspired_oxygen")),
                peep_cmh2o=self._read_optional_float(arguments, keys=("peep_cmh2o", "peep")),
                inspiratory_pressure_cmh2o=self._read_optional_float(arguments, keys=("inspiratory_pressure_cmh2o", "pressure_cmh2o")),
                respiration_rate_bpm=self._read_optional_float(arguments, keys=("respiration_rate_bpm", "rate_bpm")),
                inspiratory_period_s=self._read_optional_float(arguments, keys=("inspiratory_period_s",)),
                advance_time_seconds=monitor_seconds,
            )
        elif support_key in {"volume_control_ventilation", "volume_control"}:
            state = self._adapter.apply_volume_control_ventilation(
                fio2=self._read_optional_float(arguments, keys=("fio2", "fraction_inspired_oxygen")),
                peep_cmh2o=self._read_optional_float(arguments, keys=("peep_cmh2o", "peep")),
                tidal_volume_ml=self._read_optional_float(arguments, keys=("tidal_volume_ml", "squeeze_volume_ml")),
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

    def _handle_apply_bag_valve_mask(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        rate_bpm = self._read_optional_float(arguments, keys=("rate_bpm", "respiration_rate_bpm")) or 12.0
        volume_ml = self._read_optional_float(arguments, keys=("volume_ml", "tidal_volume_ml")) or 500.0
        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0
        state = self._adapter.apply_bag_valve_mask(
            respiration_rate_bpm=rate_bpm,
            squeeze_volume_ml=volume_ml,
            fio2=self._read_optional_float(arguments, keys=("fio2", "fraction_inspired_oxygen")),
            peep_cmh2o=self._read_optional_float(arguments, keys=("peep_cmh2o", "peep")),
            airway_adjunct=arguments.get("airway_adjunct"),
            advance_time_seconds=monitor_seconds,
        )
        return self._success(
            "apply_bag_valve_mask",
            state,
            f"BVM ventilation started at {self._fmt(rate_bpm)} bpm with {self._fmt(volume_ml)} mL squeezes.",
            previous_state=previous,
        )

    def _handle_perform_intubation(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        airway_type = str(arguments.get("airway_type") or arguments.get("kind") or "tracheal")
        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0
        state = self._adapter.set_intubation(airway_type, advance_time_seconds=monitor_seconds)
        return self._success(
            "perform_intubation",
            state,
            f"Performed {airway_type.replace('_', ' ')} intubation.",
            previous_state=previous,
        )

    def _handle_set_ventilator_tidal_volume(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        tidal_volume_ml = self._read_positive_float(arguments, keys=("volume_ml", "tidal_volume_ml"))
        runtime = self._adapter._runtime
        state = self._adapter.apply_volume_control_ventilation(
            fio2=self._read_optional_float(arguments, keys=("fio2",)) or runtime.fio2,
            peep_cmh2o=self._read_optional_float(arguments, keys=("peep_cmh2o", "peep")) or runtime.peep_cmh2o,
            tidal_volume_ml=tidal_volume_ml,
            respiration_rate_bpm=self._read_optional_float(arguments, keys=("breaths_per_min", "respiration_rate_bpm", "rate_bpm")) or runtime.ventilator_respiration_rate_bpm,
            advance_time_seconds=self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0,
        )
        return self._success(
            "set_ventilator_tidal_volume",
            state,
            f"Ventilator tidal volume set to {self._fmt(tidal_volume_ml)} mL.",
            previous_state=previous,
        )

    def _handle_set_ventilator_rate(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        breaths_per_min = self._read_positive_float(arguments, keys=("breaths_per_min", "respiration_rate_bpm", "rate_bpm"))
        runtime = self._adapter._runtime
        state = self._adapter.apply_volume_control_ventilation(
            fio2=self._read_optional_float(arguments, keys=("fio2",)) or runtime.fio2,
            peep_cmh2o=self._read_optional_float(arguments, keys=("peep_cmh2o", "peep")) or runtime.peep_cmh2o,
            tidal_volume_ml=self._read_optional_float(arguments, keys=("volume_ml", "tidal_volume_ml")) or runtime.ventilator_tidal_volume_ml,
            respiration_rate_bpm=breaths_per_min,
            advance_time_seconds=self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0,
        )
        return self._success(
            "set_ventilator_rate",
            state,
            f"Ventilator rate set to {self._fmt(breaths_per_min)} breaths/min.",
            previous_state=previous,
        )

    def _handle_set_ventilator_fio2(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        fio2 = self._read_positive_float(arguments, keys=("fraction", "fio2", "fraction_inspired_oxygen"))
        if fio2 > 1.0:
            raise ValueError("fio2 must be between 0.21 and 1.0.")
        runtime = self._adapter._runtime
        state = self._adapter.apply_volume_control_ventilation(
            fio2=fio2,
            peep_cmh2o=self._read_optional_float(arguments, keys=("peep_cmh2o", "peep")) or runtime.peep_cmh2o,
            tidal_volume_ml=self._read_optional_float(arguments, keys=("volume_ml", "tidal_volume_ml")) or runtime.ventilator_tidal_volume_ml,
            respiration_rate_bpm=self._read_optional_float(arguments, keys=("breaths_per_min", "respiration_rate_bpm", "rate_bpm")) or runtime.ventilator_respiration_rate_bpm,
            advance_time_seconds=self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0,
        )
        return self._success(
            "set_ventilator_fio2",
            state,
            f"Ventilator FiO2 set to {fio2:.2f}.",
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

    def _handle_start_pressor_infusion(self, arguments: dict[str, Any], *, tool_name: str, pressor: str) -> ToolExecution:
        previous = self._adapter.get_full_state()
        if pressor not in PulseEngineAdapter._PRESSOR_CONFIG:
            return self._failure(
                tool_name=tool_name,
                state=previous,
                code="UNSUPPORTED_BY_ENGINE",
                message=f"The local Pulse build does not expose {pressor}.",
                retryable=False,
            )
        dose = self._read_positive_float(arguments, keys=("dose_mcg_per_kg_per_min", "dose"))
        weight_kg = self._adapter.get_patient_weight_kg()
        if weight_kg is None or weight_kg <= 0:
            raise RuntimeError("Patient weight is unavailable for mcg/kg/min infusion dosing.")
        concentration_ug_per_ml = (
            self._read_optional_float(arguments, keys=("concentration_ug_per_ml",))
            or PulseEngineAdapter._PRESSOR_CONFIG[pressor]["default_concentration_ug_per_ml"]
        )
        rate_ml_per_min = (dose * weight_kg) / concentration_ug_per_ml
        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0
        state = self._adapter.set_pressor(
            pressor=pressor,
            concentration_ug_per_ml=concentration_ug_per_ml,
            rate_ml_per_min=rate_ml_per_min,
            advance_time_seconds=monitor_seconds,
        )
        return self._success(
            tool_name,
            state,
            f"Started {pressor} infusion at {dose:.2f} mcg/kg/min ({rate_ml_per_min:.2f} mL/min).",
            previous_state=previous,
        )

    def _handle_adjust_infusion_rate(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        drug_name = str(arguments.get("drug_name") or arguments.get("pressor") or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not drug_name:
            raise ValueError("drug_name is required.")
        if drug_name not in PulseEngineAdapter._PRESSOR_CONFIG:
            return self._failure(
                tool_name="adjust_infusion_rate",
                state=previous,
                code="UNSUPPORTED_BY_ENGINE",
                message=f"Only pressor infusions are adjustable right now. Unsupported infusion '{drug_name}'.",
                retryable=False,
            )
        new_dose = self._read_positive_float(arguments, keys=("new_dose", "dose_mcg_per_kg_per_min", "dose"))
        weight_kg = self._adapter.get_patient_weight_kg()
        if weight_kg is None or weight_kg <= 0:
            raise RuntimeError("Patient weight is unavailable for mcg/kg/min infusion dosing.")
        concentration_ug_per_ml = (
            self._read_optional_float(arguments, keys=("concentration_ug_per_ml",))
            or PulseEngineAdapter._PRESSOR_CONFIG[drug_name]["default_concentration_ug_per_ml"]
        )
        rate_ml_per_min = (new_dose * weight_kg) / concentration_ug_per_ml
        state = self._adapter.set_pressor(
            pressor=drug_name,
            concentration_ug_per_ml=concentration_ug_per_ml,
            rate_ml_per_min=rate_ml_per_min,
            advance_time_seconds=self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0,
        )
        return self._success(
            "adjust_infusion_rate",
            state,
            f"Adjusted {drug_name} to {new_dose:.2f} mcg/kg/min ({rate_ml_per_min:.2f} mL/min).",
            previous_state=previous,
        )

    def _handle_stop_infusion(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        drug_name = str(arguments.get("drug_name") or arguments.get("pressor") or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not drug_name:
            raise ValueError("drug_name is required.")
        if drug_name not in PulseEngineAdapter._PRESSOR_CONFIG:
            return self._failure(
                tool_name="stop_infusion",
                state=previous,
                code="UNSUPPORTED_BY_ENGINE",
                message=f"Only pressor infusions are stoppable right now. Unsupported infusion '{drug_name}'.",
                retryable=False,
            )
        state = self._adapter.set_pressor(
            pressor=drug_name,
            concentration_ug_per_ml=PulseEngineAdapter._PRESSOR_CONFIG[drug_name]["default_concentration_ug_per_ml"],
            rate_ml_per_min=0.0,
            advance_time_seconds=self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0,
        )
        return self._success("stop_infusion", state, f"Stopped {drug_name} infusion.", previous_state=previous)

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

    def _handle_perform_needle_decompression(self, arguments: dict[str, Any]) -> ToolExecution:
        result = self._handle_needle_decompression(arguments)
        result.tool_result.tool_name = "perform_needle_decompression"
        return result

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

    def _handle_perform_pericardiocentesis(self, arguments: dict[str, Any]) -> ToolExecution:
        result = self._handle_pericardiocentesis(arguments)
        result.tool_result.tool_name = "perform_pericardiocentesis"
        return result

    def _handle_perform_cpr(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        rate = self._read_optional_float(arguments, keys=("compression_rate_bpm", "rate_bpm")) or PulseEngineAdapter.DEFAULT_CPR_RATE_BPM
        depth_cm = self._read_optional_float(arguments, keys=("depth_cm",)) or PulseEngineAdapter.DEFAULT_CPR_DEPTH_CM
        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0
        state = self._adapter.perform_cpr(
            compression_rate_bpm=rate,
            depth_cm=depth_cm,
            advance_time_seconds=monitor_seconds,
        )
        return self._success(
            "perform_cpr",
            state,
            f"Performed CPR at {self._fmt(rate)} compressions/min with {self._fmt(depth_cm)} cm depth.",
            previous_state=previous,
        )

    def _handle_induce_cardiac_arrest(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        restriction = self._restricted_tool_failure("induce_cardiac_arrest", previous, arguments)
        if restriction is not None:
            return restriction
        state = self._adapter.induce_cardiac_arrest(
            advance_time_seconds=self._read_optional_float(arguments, keys=("monitor_seconds",)) or 30.0,
        )
        return self._success("induce_cardiac_arrest", state, "Induced cardiac arrest for scenario-authoring.", previous_state=previous)

    def _handle_apply_pericardial_effusion(self, arguments: dict[str, Any]) -> ToolExecution:
        previous = self._adapter.get_full_state()
        restriction = self._restricted_tool_failure("apply_pericardial_effusion", previous, arguments)
        if restriction is not None:
            return restriction
        severity = self._read_positive_float(arguments, keys=("severity",))
        if severity > 1.0:
            raise ValueError("severity must be between 0.0 and 1.0.")
        effusion_rate_ml_per_min = severity * 150.0
        state = self._adapter.set_pericardial_effusion(
            effusion_rate_ml_per_min=effusion_rate_ml_per_min,
            advance_time_seconds=self._read_optional_float(arguments, keys=("monitor_seconds",)) or 60.0,
        )
        return self._success(
            "apply_pericardial_effusion",
            state,
            f"Applied pericardial effusion with severity {severity:.2f}.",
            previous_state=previous,
        )

    def _handle_bolus_drug(
        self,
        arguments: dict[str, Any],
        *,
        tool_name: str,
        drug_key: str,
        dose_keys: tuple[str, ...],
    ) -> ToolExecution:
        previous = self._adapter.get_full_state()
        config = PulseEngineAdapter._BOLUS_DRUG_CONFIG.get(drug_key)
        if config is None:
            return self._failure(
                tool_name=tool_name,
                state=previous,
                code="UNSUPPORTED_BY_ENGINE",
                message=f"The local Pulse build does not expose {drug_key}.",
                retryable=False,
            )
        dose_input = self._read_positive_float(arguments, keys=dose_keys)
        total_dose = self._adapter.compute_drug_dose(dose=dose_input, dose_unit=config["dose_argument"])
        monitor_seconds = self._read_optional_float(arguments, keys=("monitor_seconds",)) or config["duration_s"]
        state = self._adapter.administer_substance_bolus(
            drug_name=config["substance"],
            dose=total_dose,
            dose_unit=config["dose_argument"],
            concentration_value=config["concentration_value"],
            concentration_unit=config["concentration_unit"],
            duration_s=config["duration_s"],
            advance_time_seconds=monitor_seconds,
        )
        return self._success(
            tool_name,
            state,
            f"Administered {config['substance']} bolus with input dose {dose_input:.2f} {config['dose_argument'].replace('_', '/').replace('mg', 'mg').replace('mcg', 'mcg')}.",
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

    def _handle_recommend_next_step(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        recommendation = self._recommend_next_tool(state)
        return self._success(
            "recommend_next_step",
            state,
            f"Recommended next step: {recommendation}.",
            previous_state=state,
        )

    def _handle_get_blood_gas(self, _arguments: dict[str, Any], *, tool_name: str) -> ToolExecution:
        return self._handle_delayed_diagnostic(
            tool_name=tool_name,
            diagnostic_key="order_arterial_blood_gas",
            label="ABG",
            render_result=lambda state: (
                f"ABG pH {self._fmt(state.abg_result.ph, precision=3)}, PaO2 "
                f"{self._fmt(state.abg_result.partial_pressure_of_oxygen_mmhg)} mmHg, PaCO2 "
                f"{self._fmt(state.abg_result.partial_pressure_of_carbon_dioxide_mmhg)} mmHg, lactate "
                f"{self._fmt(state.abg_result.lactate_mg_per_dl)} mg/dL."
            ),
        )

    def _handle_get_cbc(self, _arguments: dict[str, Any], *, tool_name: str) -> ToolExecution:
        return self._handle_delayed_diagnostic(
            tool_name=tool_name,
            diagnostic_key="order_complete_blood_count",
            label="CBC",
            render_result=lambda state: (
                f"CBC hemoglobin {self._fmt(state.cbc_result.hemoglobin_g_per_dl)} g/dL, hematocrit "
                f"{self._fmt(state.cbc_result.hematocrit_fraction, precision=3)}, WBC "
                f"{self._fmt(state.cbc_result.white_blood_cell_count_per_u_l)} /uL."
            ),
        )

    def _handle_get_bmp(self, _arguments: dict[str, Any], *, tool_name: str) -> ToolExecution:
        return self._handle_delayed_diagnostic(
            tool_name=tool_name,
            diagnostic_key="order_basic_metabolic_panel",
            label="BMP",
            render_result=lambda state: (
                f"BMP sodium {self._fmt(state.bmp_result.sodium_mmol_per_l)} mmol/L, potassium "
                f"{self._fmt(state.bmp_result.potassium_mmol_per_l)} mmol/L, creatinine "
                f"{self._fmt(state.bmp_result.creatinine_mg_per_dl)} mg/dL, glucose "
                f"{self._fmt(state.bmp_result.glucose_mg_per_dl)} mg/dL."
            ),
        )

    def _handle_order_point_of_care_ultrasound(self, arguments: dict[str, Any]) -> ToolExecution:
        region = str(arguments.get("region") or "cardiac").strip().lower().replace("-", "_").replace(" ", "_")
        diagnostic_key = f"order_point_of_care_ultrasound:{region}"
        return self._handle_delayed_diagnostic(
            tool_name="order_point_of_care_ultrasound",
            diagnostic_key=diagnostic_key,
            label=f"POCUS ({region})",
            delay_seconds=PulseEngineAdapter.DIAGNOSTIC_DELAYS_S["order_point_of_care_ultrasound"],
            render_result=lambda _state, region=region: self._adapter.get_ultrasound_summary(region),
        )

    def _handle_auscultate_chest(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        if state.breath_sounds == "present bilateral":
            message = "Auscultation: bilateral breath sounds present."
        else:
            message = f"Auscultation: breath sounds are {state.breath_sounds}."
        return self._success("auscultate_chest", state, message, previous_state=state)

    def _handle_assess_consciousness_level(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        consciousness = self._adapter.get_consciousness_summary()
        message = (
            f"Consciousness level {consciousness['mental_status']} with approximate GCS "
            f"{self._fmt(float(consciousness['gcs_equivalent']))}."
        )
        return self._success("assess_consciousness_level", state, message, previous_state=state)

    def _handle_check_pain_level(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        pain_score = self._adapter.get_pain_score_0_to_10()
        message = f"Estimated pain score {pain_score:.1f}/10."
        return self._success("check_pain_level", state, message, previous_state=state)

    def _handle_measure_core_temperature(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        message = f"Core temperature {self._fmt(state.core_temperature_c)} C."
        return self._success("measure_core_temperature", state, message, previous_state=state)

    def _handle_check_end_tidal_co2(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        message = f"End tidal CO2 {self._fmt(state.etco2_mmhg)} mmHg."
        return self._success("check_end_tidal_co2", state, message, previous_state=state)

    def _handle_calculate_shock_index(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        shock = self._adapter.get_shock_assessment()
        message = f"Shock index {self._fmt(shock['shock_index'], precision=2)} with class {shock['shock_class']}."
        return self._success("calculate_shock_index", state, message, previous_state=state)

    def _handle_assess_urine_output(self, _arguments: dict[str, Any]) -> ToolExecution:
        state = self._adapter.get_full_state()
        shock = self._adapter.get_shock_assessment()
        message = f"Estimated urine output {self._fmt(shock['urine_output_ml_per_hr'])} mL/hr."
        return self._success("assess_urine_output", state, message, previous_state=state)

    def _recommend_next_tool(self, state: PatientState) -> str:
        if "possible_tension_pneumothorax" in state.active_alerts or "unilateral_absent_breath_sounds" in state.active_alerts:
            return "perform_needle_decompression"
        if "possible_cardiac_tamponade" in state.active_alerts:
            return "perform_pericardiocentesis"
        if state.active_hemorrhages:
            limb_sites = {"left_arm", "right_arm", "left_leg", "right_leg"}
            if any(site in limb_sites for site in state.active_hemorrhages):
                return "apply_tourniquet"
            return "apply_direct_pressure"
        if state.spo2 is not None and state.spo2 < 0.88 and state.airway_support not in {
            "bag_valve_mask",
            "pressure_control_ventilation",
            "volume_control_ventilation",
        }:
            return "apply_bag_valve_mask"
        if state.spo2 is not None and state.spo2 < 0.92:
            return "apply_nonrebreather_mask"
        if state.mean_arterial_pressure_mmhg is not None and state.mean_arterial_pressure_mmhg < 65:
            if any(key in state.active_infusions for key in {"blood", "packed_rbc", "saline"}):
                return "start_norepinephrine_infusion"
            return "administer_crystalloid_bolus"
        if any(key.startswith("order_arterial_blood_gas") for key in state.ready_diagnostics):
            return "order_arterial_blood_gas"
        if state.mental_status in {"pain", "unresponsive"} and state.airway_support is None:
            return "perform_intubation"
        return "get_vitals"

    def _handle_delayed_diagnostic(
        self,
        *,
        tool_name: str,
        diagnostic_key: str,
        label: str,
        render_result: Callable[[PatientState], str],
        delay_seconds: int | None = None,
    ) -> ToolExecution:
        previous = self._adapter.get_full_state()
        if diagnostic_key in previous.ready_diagnostics:
            return self._success(tool_name, previous, render_result(previous), previous_state=previous)

        if diagnostic_key in previous.pending_diagnostics:
            remaining = previous.pending_diagnostics[diagnostic_key]
            return self._success(
                tool_name,
                previous,
                f"{label} is pending. {remaining} simulated seconds remaining before results are ready.",
                previous_state=previous,
            )

        state = self._adapter.schedule_diagnostic(
            diagnostic_key,
            delay_seconds=delay_seconds or self._default_diagnostic_delay(diagnostic_key),
        )
        remaining = state.pending_diagnostics.get(diagnostic_key)
        return self._success(
            tool_name,
            state,
            f"Ordered {label}. Results will be ready after about {remaining} simulated seconds.",
            previous_state=previous,
        )

    def _restricted_tool_failure(
        self,
        tool_name: str,
        state: PatientState,
        arguments: dict[str, Any],
    ) -> ToolExecution | None:
        if bool(arguments.get("admin_override")):
            return None
        return self._failure(
            tool_name=tool_name,
            state=state,
            code="RESTRICTED_TOOL",
            message=f"{tool_name} is restricted to scenario-authoring workflows. Pass admin_override=true to use it deliberately.",
            retryable=False,
        )

    def _resolve_bleeding_site(self, state: PatientState, arguments: dict[str, Any]) -> str | None:
        if not state.active_hemorrhages:
            return None
        site = self._normalize_site(arguments.get("site") or arguments.get("compartment"))
        if site:
            if site not in state.active_hemorrhages:
                raise ValueError(f"'{site}' is not an active hemorrhage site.")
            return site
        if len(state.active_hemorrhages) == 1:
            return next(iter(state.active_hemorrhages))
        raise ValueError("Multiple active hemorrhages are present; specify a site.")

    @staticmethod
    def _normalize_site(value: object) -> str:
        if value is None:
            return ""
        return str(value).strip().lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _suggest_oxygen_device(state: PatientState) -> str:
        if state.spo2 is not None and state.spo2 < 0.9:
            return "non_rebreather_mask"
        if state.spo2 is not None and state.spo2 < 0.95:
            return "simple_mask"
        return "nasal_cannula"

    @staticmethod
    def _default_oxygen_flow(device: str) -> float:
        return {
            "nasal_cannula": 4.0,
            "simple_mask": 8.0,
            "non_rebreather_mask": 15.0,
        }[device]

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

    @staticmethod
    def _suggest_needle_side(state: PatientState) -> str:
        if "absent left" in state.breath_sounds:
            return "left"
        if "absent right" in state.breath_sounds:
            return "right"
        return "left"

    @staticmethod
    def _default_diagnostic_delay(diagnostic_key: str) -> int:
        base_key = diagnostic_key.split(":", 1)[0]
        return PulseEngineAdapter.DIAGNOSTIC_DELAYS_S.get(base_key, 120)

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
