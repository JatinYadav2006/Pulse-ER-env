"""Backend adapters for swapping mock and real Pulse runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
import random

from ..models import (
    INITIAL_TOOL_NAMES,
    EnvironmentResponse,
    ObservationMetadata,
    PatientState,
    PulsePhysiologyObservation,
    ToolAction,
    ToolError,
    ToolResult,
)
from ..runtime_effects import NoisyObservation, TimePressureMechanic
from ..rewards import compute_reward
from ..tool_catalog import (
    KNOWN_TOOL_NAMES,
    ToolValidationError,
    canonicalize_tool_name,
    coerce_boolean_argument,
    validate_tool_arguments,
)
from .mock_scenarios import DEFAULT_MOCK_SCENARIO_ID, MOCK_SCENARIOS, MockScenarioDefinition


MOCK_READ_ONLY_TOOLS = {
    "get_vitals",
    "summarize_state",
    "check_deterioration",
    "recommend_next_step",
    "get_respiratory_status",
    "get_blood_gas",
    "get_cbc",
    "get_bmp",
}

MOCK_DIAGNOSTIC_DELAYS = {
    "get_blood_gas": 120,
    "get_cbc": 240,
    "get_bmp": 300,
}

MOCK_EXTENDED_INTERVENTION_EFFECTS = {
    "give_oxygen": {
        "baseline_stable": {},
        "respiratory_distress": {"spo2": 0.04, "respiration_rate_bpm": -3.0, "heart_rate_bpm": -2.0},
        "hemorrhagic_shock": {"spo2": 0.02, "respiration_rate_bpm": -1.0, "heart_rate_bpm": -1.0},
    },
    "give_fluids": {
        "baseline_stable": {"heart_rate_bpm": 2.0},
        "respiratory_distress": {"systolic_bp_mmhg": 0.5, "heart_rate_bpm": 1.0},
        "hemorrhagic_shock": {"systolic_bp_mmhg": 8.0, "diastolic_bp_mmhg": 4.0, "heart_rate_bpm": -4.0, "blood_volume_ml": 300.0},
    },
    "control_bleeding": {
        "baseline_stable": {},
        "respiratory_distress": {},
        "hemorrhagic_shock": {"systolic_bp_mmhg": 5.0, "diastolic_bp_mmhg": 2.5, "heart_rate_bpm": -3.0, "blood_volume_ml": 150.0},
    },
    "position_patient": {
        "baseline_stable": {},
        "respiratory_distress": {"spo2": 0.01, "respiration_rate_bpm": -1.0},
        "hemorrhagic_shock": {"systolic_bp_mmhg": 2.0, "diastolic_bp_mmhg": 1.0, "respiration_rate_bpm": -1.0},
    },
    "airway_support": {
        "baseline_stable": {"heart_rate_bpm": 1.0},
        "respiratory_distress": {"spo2": 0.04, "respiration_rate_bpm": -4.0, "heart_rate_bpm": -2.0},
        "hemorrhagic_shock": {"spo2": 0.01, "respiration_rate_bpm": -1.0, "heart_rate_bpm": -0.5},
    },
    "give_pressor": {
        "baseline_stable": {"systolic_bp_mmhg": 0.5, "diastolic_bp_mmhg": 0.2, "heart_rate_bpm": 3.0},
        "respiratory_distress": {"systolic_bp_mmhg": 1.0, "diastolic_bp_mmhg": 0.5, "heart_rate_bpm": 2.0},
        "hemorrhagic_shock": {"systolic_bp_mmhg": 4.0, "diastolic_bp_mmhg": 2.0, "heart_rate_bpm": -1.0},
    },
    "needle_decompression": {
        "baseline_stable": {"systolic_bp_mmhg": -1.0, "heart_rate_bpm": 1.0},
        "respiratory_distress": {"spo2": 0.06, "respiration_rate_bpm": -6.0, "heart_rate_bpm": -4.0},
        "hemorrhagic_shock": {"systolic_bp_mmhg": -1.0, "heart_rate_bpm": 0.5},
    },
    "pericardiocentesis": {
        "baseline_stable": {"systolic_bp_mmhg": -2.0, "diastolic_bp_mmhg": -1.0, "heart_rate_bpm": 2.0},
        "respiratory_distress": {"systolic_bp_mmhg": -1.0, "diastolic_bp_mmhg": -0.5, "heart_rate_bpm": 1.0},
        "hemorrhagic_shock": {"systolic_bp_mmhg": -1.0, "diastolic_bp_mmhg": -0.5, "heart_rate_bpm": 1.0},
    },
}


class PatientBackend(ABC):
    """Stable interface between Person 2's stack and the backend runtime."""

    @abstractmethod
    def reset(self, scenario_id: str | None = None, **kwargs: object) -> EnvironmentResponse:
        """Reset the environment and return the initial response."""

    @abstractmethod
    def step(self, action: ToolAction) -> EnvironmentResponse:
        """Apply one action and return the next response."""

    @abstractmethod
    def get_state(self) -> PatientState:
        """Return the latest patient state."""


class MockPulseAdapter(PatientBackend):
    """Deterministic backend used by Person 2 before real Pulse integration exists."""

    def __init__(
        self,
        default_scenario_id: str = DEFAULT_MOCK_SCENARIO_ID,
        *,
        observation_noise_level: float = 0.0,
        time_pressure_enabled: bool = False,
        time_pressure_onset_s: float = 180.0,
        time_pressure_escalation_per_minute: float = 0.15,
        seed: int | None = None,
    ):
        self._default_scenario_id = default_scenario_id
        self._scenario: MockScenarioDefinition | None = None
        self._state: PatientState | None = None
        self._step_count = 0
        self._active_supports: set[str] = set()
        self._tool_counts: dict[str, int] = {}
        self._last_tool_name: str | None = None
        self._same_tool_called_consecutively = 0
        self._rng = random.Random(seed)
        self._observation_noise = NoisyObservation(observation_noise_level)
        self._time_pressure = TimePressureMechanic(
            enabled=time_pressure_enabled,
            onset_s=time_pressure_onset_s,
            escalation_per_minute=time_pressure_escalation_per_minute,
        )
        self._episode_observation_rng = random.Random(self._rng.random())

    def reset(self, scenario_id: str | None = None, **kwargs: object) -> EnvironmentResponse:
        selected_scenario_id = scenario_id or self._default_scenario_id
        if selected_scenario_id not in MOCK_SCENARIOS:
            valid = ", ".join(sorted(MOCK_SCENARIOS))
            raise ValueError(
                f"Unknown mock scenario_id '{selected_scenario_id}'. Expected one of: {valid}"
            )
        scenario = MOCK_SCENARIOS[selected_scenario_id]

        self._scenario = scenario
        self._configure_runtime_effects(kwargs)
        self._state = self._refresh_state(scenario.initial_state.model_copy(deep=True))
        self._step_count = 0
        self._active_supports = set()
        self._tool_counts = {}
        self._last_tool_name = None
        self._same_tool_called_consecutively = 0
        self._episode_observation_rng = random.Random(self._rng.random())

        return self._build_response(
            reward=0.0,
            tool_result=ToolResult(
                tool_name="load_scenario",
                success=True,
                message=f"Scenario '{scenario.scenario_id}' loaded.",
                state_changed=True,
                changed_fields=list(self._state.model_dump().keys()),
            ),
        )

    def step(self, action: ToolAction) -> EnvironmentResponse:
        if self._state is None or self._scenario is None:
            return self._error_response(
                code="NOT_INITIALIZED",
                message="Call reset() before step().",
                retryable=True,
                tool_name=action.tool_name,
            )

        canonical_tool_name = canonicalize_tool_name(action.tool_name, allowed_tools=list(KNOWN_TOOL_NAMES))
        action = action.model_copy(update={"tool_name": canonical_tool_name})

        if action.tool_name not in KNOWN_TOOL_NAMES:
            return self._error_response(
                code="UNKNOWN_TOOL",
                message=f"Unsupported tool '{action.tool_name}'.",
                retryable=False,
                tool_name=action.tool_name,
            )

        try:
            normalized_arguments = validate_tool_arguments(
                action.tool_name,
                action.arguments,
                allowed_tools=KNOWN_TOOL_NAMES,
            )
        except ToolValidationError as exc:
            return self._error_response(
                code="INVALID_ARGUMENT",
                message=str(exc),
                retryable=False,
                tool_name=action.tool_name,
            )

        action = action.model_copy(update={"arguments": normalized_arguments})

        previous_state = self._state.model_copy(deep=True)
        self._step_count += 1

        if action.tool_name == "advance_time":
            result = self._advance_time(action)
        elif action.tool_name in MOCK_READ_ONLY_TOOLS:
            result = self._read_only_tool(action.tool_name)
        else:
            result = self._apply_intervention(action)

        if result.error is not None:
            return result

        self._state = self._refresh_state(self._state)
        changed_fields = self._changed_fields(previous_state, self._state)
        tool_usage_count = self._tool_counts.get(action.tool_name, 0) + 1
        if self._last_tool_name == action.tool_name:
            self._same_tool_called_consecutively += 1
        else:
            self._last_tool_name = action.tool_name
            self._same_tool_called_consecutively = 1
        self._tool_counts[action.tool_name] = tool_usage_count
        reward = compute_reward(
            previous_state,
            self._state,
            action.tool_name,
            self._scenario.recommended_actions,
            tool_usage_count=tool_usage_count,
            same_tool_called_consecutively=self._same_tool_called_consecutively,
            state_changed=bool(changed_fields),
            time_pressure_multiplier=self._current_time_pressure_multiplier(self._state),
        ).total

        tool_result = result.tool_result or ToolResult(
            tool_name=action.tool_name,
            success=True,
            message=f"{action.tool_name} executed.",
            state_changed=bool(changed_fields),
            changed_fields=changed_fields,
        )
        tool_result.changed_fields = changed_fields
        tool_result.state_changed = bool(changed_fields)

        return self._build_response(reward=reward, tool_result=tool_result)

    def get_state(self) -> PatientState:
        if self._state is None:
            raise RuntimeError("MockPulseAdapter has not been reset yet.")
        return self._state.model_copy(deep=True)

    def _advance_time(self, action: ToolAction) -> EnvironmentResponse:
        assert self._state is not None
        assert self._scenario is not None

        seconds = float(action.arguments.get("seconds", 30))
        if seconds <= 0:
            return self._error_response(
                code="INVALID_ARGUMENT",
                message="seconds must be greater than 0",
                retryable=False,
                tool_name=action.tool_name,
            )

        scale = seconds / 30.0
        updates = self._state.model_dump()
        deterioration_multiplier = self._current_time_pressure_multiplier(self._state)

        for field_name, delta in self._scenario.deterioration_per_30s.items():
            adjusted_delta = self._deterioration_delta(field_name, delta) * deterioration_multiplier
            current_value = updates.get(field_name)
            if current_value is None:
                continue
            updates[field_name] = current_value + adjusted_delta * scale

        updates["sim_time_s"] = self._state.sim_time_s + seconds
        pending_diagnostics = dict(self._state.pending_diagnostics)
        ready_diagnostics = list(self._state.ready_diagnostics)
        for tool_name, remaining_seconds in list(pending_diagnostics.items()):
            remaining_after_step = max(0, int(remaining_seconds - seconds))
            if remaining_after_step <= 0:
                pending_diagnostics.pop(tool_name, None)
                if tool_name not in ready_diagnostics:
                    ready_diagnostics.append(tool_name)
            else:
                pending_diagnostics[tool_name] = remaining_after_step
        updates["pending_diagnostics"] = pending_diagnostics
        updates["ready_diagnostics"] = ready_diagnostics
        self._state = PatientState(**updates)

        return self._build_response(
            reward=0.0,
            tool_result=ToolResult(
                tool_name=action.tool_name,
                success=True,
                message=f"Advanced simulation by {seconds:.0f} seconds.",
                state_changed=True,
                changed_fields=[],
            ),
        )

    def _apply_intervention(self, action: ToolAction) -> EnvironmentResponse:
        assert self._state is not None
        assert self._scenario is not None

        if action.tool_name == "give_pressor" and action.arguments.get("stop") is True:
            updates = self._state.model_dump()
            self._apply_tool_side_effects(action, updates, effect_scale=0.0)
            self._active_supports.discard("give_pressor")
            self._state = PatientState(**updates)
            return self._build_response(
                reward=0.0,
                tool_result=ToolResult(
                    tool_name=action.tool_name,
                    success=True,
                    message="Vasopressor support stopped.",
                    state_changed=True,
                    changed_fields=[],
                ),
            )

        effects = self._scenario.tool_effects.get(action.tool_name)
        if effects is None:
            effects = MOCK_EXTENDED_INTERVENTION_EFFECTS.get(action.tool_name, {}).get(self._scenario.scenario_id)
        if effects is None:
            return self._error_response(
                code="UNSUPPORTED_IN_SCENARIO",
                message=f"{action.tool_name} is not modeled for scenario '{self._scenario.scenario_id}'.",
                retryable=False,
                tool_name=action.tool_name,
            )

        updates = self._state.model_dump()
        effect_scale = self._intervention_scale(action.tool_name)
        for field_name, delta in effects.items():
            current_value = updates.get(field_name)
            if current_value is None:
                continue
            updates[field_name] = current_value + (delta * effect_scale)

        self._apply_tool_side_effects(action, updates, effect_scale)
        self._active_supports.add(action.tool_name)
        self._state = PatientState(**updates)

        return self._build_response(
            reward=0.0,
            tool_result=ToolResult(
                tool_name=action.tool_name,
                success=True,
                message=self._tool_message(action.tool_name),
                state_changed=True,
                changed_fields=[],
            ),
        )

    def _intervention_scale(self, tool_name: str) -> float:
        assert self._state is not None

        alerts = set(self._state.active_alerts)
        scale = 1.0
        intervention_multiplier = self._time_pressure.intervention_effectiveness_multiplier(
            sim_time_s=self._state.sim_time_s,
            injury_severity=self._scenario.injury_severity,
            unstable=self._is_state_unstable(self._state),
        )

        if tool_name == "control_bleeding":
            if "blood_loss" in alerts:
                scale = 1.0
            elif "hypotension" in alerts:
                scale = 0.5
            else:
                scale = 0.15
        elif tool_name == "give_fluids":
            if {"hypotension", "blood_loss"} & alerts:
                scale = 1.0
            elif "tachycardia" in alerts:
                scale = 0.6
            else:
                scale = 0.2
        elif tool_name == "give_oxygen":
            if {"hypoxemia", "tachypnea"} & alerts:
                scale = 1.0
            elif "tachycardia" in alerts:
                scale = 0.4
            else:
                scale = 0.15
        elif tool_name == "position_patient":
            if {"tachypnea", "hypotension"} & alerts:
                scale = 1.0
            else:
                scale = 0.25
        elif tool_name == "airway_support":
            if {"hypoxemia", "tachypnea"} & alerts:
                scale = 1.0
            else:
                scale = 0.2

        if tool_name in self._active_supports:
            scale *= 0.7

        return scale * intervention_multiplier

    def _read_only_tool(self, tool_name: str) -> EnvironmentResponse:
        assert self._state is not None
        assert self._scenario is not None

        if tool_name == "summarize_state":
            message = (
                f"{self._scenario.description} HR {self._state.heart_rate_bpm:.0f}, "
                f"BP {self._state.systolic_bp_mmhg:.0f}/{self._state.diastolic_bp_mmhg:.0f}, "
                f"SpO2 {self._state.spo2:.2f}."
            )
        elif tool_name == "check_deterioration":
            message = "Deterioration ongoing." if self._state.active_alerts else "Patient currently stable."
        elif tool_name == "recommend_next_step":
            message = f"Recommended next step: {self._scenario.recommended_actions[0]}."
        elif tool_name == "get_respiratory_status":
            message = (
                f"Breath sounds {self._state.breath_sounds}, SpO2 {self._state.spo2:.2f}, "
                f"RR {self._state.respiration_rate_bpm:.0f}, airway support {self._state.airway_support or 'none'}."
            )
        elif tool_name in MOCK_DIAGNOSTIC_DELAYS:
            message = self._handle_diagnostic_read(tool_name)
        else:
            message = "Current vitals retrieved."

        return self._build_response(
            reward=0.0,
            tool_result=ToolResult(
                tool_name=tool_name,
                success=True,
                message=message,
                state_changed=False,
                changed_fields=[],
            ),
        )

    def _build_response(
        self,
        reward: float,
        tool_result: ToolResult | None = None,
        error: ToolError | None = None,
    ) -> EnvironmentResponse:
        assert self._state is not None
        available_tools = self._available_tools()
        observed_state, runtime_metadata = self._build_observed_state()

        return EnvironmentResponse(
            observation=PulsePhysiologyObservation.from_patient_state(
                observed_state,
                reward=reward,
                available_tools=available_tools,
                tool_result=tool_result,
                error=error,
                metadata={
                    "step_count": self._step_count,
                    **runtime_metadata,
                },
            ),
            reward=reward,
            done=observed_state.done,
            metadata=ObservationMetadata(
                step_count=self._step_count,
                available_tools=available_tools,
            ),
            tool_result=tool_result,
            error=error,
        )

    def _error_response(
        self,
        code: str,
        message: str,
        retryable: bool,
        tool_name: str,
    ) -> EnvironmentResponse:
        state = self._state or MOCK_SCENARIOS[self._default_scenario_id].initial_state
        available_tools = self._available_tools()
        if self._state is not None:
            observed_state, runtime_metadata = self._build_observed_state()
        else:
            observed_state = state
            runtime_metadata = {
                "observation_noise": {
                    "enabled": self._observation_noise.config.enabled,
                    "noise_level": round(self._observation_noise.config.normalized_level, 3),
                    "masked_fields": [],
                    "perturbed_fields": [],
                },
                "time_pressure": self._time_pressure.as_metadata(
                    sim_time_s=float(state.sim_time_s or 0.0),
                    injury_severity=self._scenario.injury_severity if self._scenario is not None else 0.0,
                    unstable=self._is_state_unstable(state),
                ),
            }

        return EnvironmentResponse(
            observation=PulsePhysiologyObservation.from_patient_state(
                observed_state,
                reward=-1.0,
                available_tools=available_tools,
                error=ToolError(code=code, message=message, retryable=retryable),
                metadata={
                    "step_count": self._step_count,
                    **runtime_metadata,
                },
            ),
            reward=-1.0,
            done=observed_state.done,
            metadata=ObservationMetadata(
                step_count=self._step_count,
                available_tools=available_tools,
            ),
            tool_result=ToolResult(
                tool_name=tool_name,
                success=False,
                message=message,
                state_changed=False,
                changed_fields=[],
            ),
            error=ToolError(code=code, message=message, retryable=retryable),
        )

    def _deterioration_delta(self, field_name: str, delta: float) -> float:
        if self._scenario is None:
            return delta

        if self._scenario.scenario_id == "respiratory_distress":
            if field_name == "spo2" and "give_oxygen" in self._active_supports:
                return delta * 0.25
            if field_name == "respiration_rate_bpm" and "airway_support" in self._active_supports:
                return delta * 0.4
        if self._scenario.scenario_id == "hemorrhagic_shock":
            if field_name == "blood_volume_ml" and "control_bleeding" in self._active_supports:
                return delta * 0.1
            if field_name in {"systolic_bp_mmhg", "diastolic_bp_mmhg"} and "give_fluids" in self._active_supports:
                if "control_bleeding" in self._active_supports or "position_patient" in self._active_supports:
                    return delta * 0.15
                return delta * 0.4
            if field_name in {"systolic_bp_mmhg", "diastolic_bp_mmhg"} and "control_bleeding" in self._active_supports:
                return delta * 0.6
            if field_name == "heart_rate_bpm":
                if {"control_bleeding", "give_fluids"} <= self._active_supports:
                    return delta * 0.25
                if "control_bleeding" in self._active_supports or "give_fluids" in self._active_supports:
                    return delta * 0.5
            if field_name == "respiration_rate_bpm":
                if "give_oxygen" in self._active_supports and "position_patient" in self._active_supports:
                    return delta * 0.35
                if "give_oxygen" in self._active_supports or "position_patient" in self._active_supports:
                    return delta * 0.6
            if field_name == "spo2" and "give_oxygen" in self._active_supports:
                return delta * 0.3
        return delta

    def _apply_tool_side_effects(
        self,
        action: ToolAction,
        updates: dict,
        effect_scale: float,
    ) -> None:
        tool_name = action.tool_name
        arguments = action.arguments

        if tool_name == "give_oxygen":
            updates["oxygen_device"] = str(arguments.get("device") or "nasal_cannula")
            updates["oxygen_flow_lpm"] = float(arguments.get("flow_lpm", 15))
        elif tool_name == "position_patient":
            updates["position"] = str(arguments.get("position") or updates.get("position") or "upright")
        elif tool_name == "airway_support":
            requested_mode = str(arguments.get("mode") or arguments.get("support_type") or "auto")
            normalized_mode = requested_mode.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized_mode in {"auto", "basic", "default", "standard", "support", "airway_support"}:
                updates["airway_support"] = self._suggest_airway_support_mode()
            else:
                updates["airway_support"] = normalized_mode
        elif tool_name == "give_fluids":
            active_infusions = dict(updates.get("active_infusions") or {})
            fluid_name = str(arguments.get("fluid_type") or arguments.get("fluid") or "saline")
            active_infusions[fluid_name] = float(arguments.get("rate_ml_per_min", 100))
            updates["active_infusions"] = active_infusions
        elif tool_name == "give_pressor":
            active_infusions = dict(updates.get("active_infusions") or {})
            pressor_name = str(arguments.get("pressor") or arguments.get("agent") or "norepinephrine")
            if arguments.get("stop") is True:
                active_infusions.pop(pressor_name, None)
            else:
                active_infusions[pressor_name] = float(arguments.get("rate_ml_per_min", 5))
            updates["active_infusions"] = active_infusions
        elif tool_name == "needle_decompression":
            updates["breath_sounds"] = "present bilateral"
        elif tool_name == "pericardiocentesis":
            updates["active_alerts"] = [
                alert
                for alert in updates.get("active_alerts", [])
                if alert != "tamponade"
            ]

        if tool_name == "needle_decompression" and self._scenario is not None:
            if self._scenario.scenario_id == "respiratory_distress":
                updates["spo2"] = min(1.0, float(updates.get("spo2") or 0.0) + (0.02 * effect_scale))

    def _handle_diagnostic_read(self, tool_name: str) -> str:
        assert self._state is not None
        if tool_name in self._state.ready_diagnostics:
            return self._diagnostic_result_message(tool_name)
        if tool_name in self._state.pending_diagnostics:
            remaining = self._state.pending_diagnostics[tool_name]
            return f"{tool_name} is pending. {remaining} simulated seconds remaining before results are ready."

        updates = self._state.model_dump()
        pending_diagnostics = dict(self._state.pending_diagnostics)
        pending_diagnostics[tool_name] = MOCK_DIAGNOSTIC_DELAYS[tool_name]
        updates["pending_diagnostics"] = pending_diagnostics
        self._state = PatientState(**updates)
        return (
            f"Ordered {tool_name}. Results will be ready after about "
            f"{MOCK_DIAGNOSTIC_DELAYS[tool_name]} simulated seconds."
        )

    def _diagnostic_result_message(self, tool_name: str) -> str:
        assert self._state is not None
        updates = self._state.model_dump()
        if tool_name == "get_blood_gas":
            abg_result = self._build_abg_result()
            updates["abg_result"] = abg_result.model_dump()
            self._state = PatientState(**updates)
            return (
                f"ABG pH {abg_result.ph:.3f}, PaO2 {abg_result.partial_pressure_of_oxygen_mmhg:.1f} mmHg, "
                f"PaCO2 {abg_result.partial_pressure_of_carbon_dioxide_mmhg:.1f} mmHg, "
                f"lactate {abg_result.lactate_mg_per_dl:.1f} mg/dL."
            )
        if tool_name == "get_cbc":
            cbc_result = self._build_cbc_result()
            updates["cbc_result"] = cbc_result.model_dump()
            self._state = PatientState(**updates)
            return (
                f"CBC hemoglobin {cbc_result.hemoglobin_g_per_dl:.1f} g/dL, "
                f"hematocrit {cbc_result.hematocrit_fraction:.3f}, "
                f"WBC {cbc_result.white_blood_cell_count_per_u_l:.0f} /uL."
            )
        bmp_result = self._build_bmp_result()
        updates["bmp_result"] = bmp_result.model_dump()
        self._state = PatientState(**updates)
        return (
            f"BMP sodium {bmp_result.sodium_mmol_per_l:.1f} mmol/L, "
            f"potassium {bmp_result.potassium_mmol_per_l:.1f} mmol/L, "
            f"creatinine {bmp_result.creatinine_mg_per_dl:.1f} mg/dL, "
            f"glucose {bmp_result.glucose_mg_per_dl:.1f} mg/dL."
        )

    def _build_abg_result(self):
        assert self._state is not None
        from ..patient_state import ArterialBloodGasResult

        spo2 = float(self._state.spo2 or 0.95)
        systolic = float(self._state.systolic_bp_mmhg or 110.0)
        ph = max(7.10, min(7.45, 7.40 - max(0.0, (95.0 - systolic) / 200.0)))
        pao2 = max(45.0, min(110.0, 40.0 + spo2 * 60.0))
        paco2 = max(28.0, min(60.0, 40.0 + max(0.0, (24.0 - float(self._state.respiration_rate_bpm or 16.0)) * 0.8)))
        lactate = max(8.0, min(40.0, 10.0 + max(0.0, (100.0 - systolic) * 0.15)))
        return ArterialBloodGasResult(
            ph=round(ph, 3),
            partial_pressure_of_oxygen_mmhg=round(pao2, 1),
            partial_pressure_of_carbon_dioxide_mmhg=round(paco2, 1),
            oxygen_saturation=round(spo2, 3),
            bicarbonate_meq_per_l=24.0,
            lactate_mg_per_dl=round(lactate, 1),
            base_excess_meq_per_l=-2.0 if systolic < 95.0 else 0.0,
            base_deficit_meq_per_l=2.0 if systolic < 95.0 else 0.0,
        )

    def _build_cbc_result(self):
        assert self._state is not None
        from ..patient_state import CompleteBloodCountResult

        blood_volume = float(self._state.blood_volume_ml or 5400.0)
        hemoglobin = max(7.5, min(15.0, 15.0 - max(0.0, (5400.0 - blood_volume) / 350.0)))
        hematocrit = max(0.24, min(0.45, hemoglobin / 33.0))
        return CompleteBloodCountResult(
            hemoglobin_g_per_dl=round(hemoglobin, 1),
            hematocrit_fraction=round(hematocrit, 3),
            white_blood_cell_count_per_u_l=9000.0 if self._state.active_alerts else 7000.0,
            platelet_count_per_u_l=250000.0,
            red_blood_cell_count_per_u_l=4800000.0,
        )

    def _build_bmp_result(self):
        assert self._state is not None
        from ..patient_state import BasicMetabolicPanelResult

        systolic = float(self._state.systolic_bp_mmhg or 110.0)
        return BasicMetabolicPanelResult(
            sodium_mmol_per_l=142.0 if systolic >= 95.0 else 145.0,
            potassium_mmol_per_l=3.8 if systolic >= 95.0 else 4.2,
            calcium_mmol_per_l=2.2,
            creatinine_mg_per_dl=1.0 if systolic >= 95.0 else 1.4,
            glucose_mg_per_dl=105.0 if not self._state.active_alerts else 128.0,
        )

    def _changed_fields(self, previous_state: PatientState, new_state: PatientState) -> list[str]:
        changed_fields: list[str] = []
        for field_name in new_state.model_fields:
            if getattr(previous_state, field_name) != getattr(new_state, field_name):
                changed_fields.append(field_name)
        return changed_fields

    def _refresh_state(self, state: PatientState) -> PatientState:
        updates = state.model_dump()
        updates["spo2"] = max(0.5, min(1.0, updates["spo2"]))
        updates["heart_rate_bpm"] = max(20.0, updates["heart_rate_bpm"])
        updates["systolic_bp_mmhg"] = max(40.0, updates["systolic_bp_mmhg"])
        updates["diastolic_bp_mmhg"] = max(20.0, updates["diastolic_bp_mmhg"])
        updates["respiration_rate_bpm"] = max(4.0, updates["respiration_rate_bpm"])
        if updates["blood_volume_ml"] is not None:
            updates["blood_volume_ml"] = max(2500.0, updates["blood_volume_ml"])

        if updates["systolic_bp_mmhg"] is not None and updates["diastolic_bp_mmhg"] is not None:
            updates["mean_arterial_pressure_mmhg"] = (
                updates["systolic_bp_mmhg"] + 2 * updates["diastolic_bp_mmhg"]
            ) / 3.0
        if updates["heart_rate_bpm"] is not None and updates["systolic_bp_mmhg"] not in (None, 0):
            updates["shock_index"] = updates["heart_rate_bpm"] / updates["systolic_bp_mmhg"]
        if self._scenario is not None and self._scenario.scenario_id == "respiratory_distress":
            updates["breath_sounds"] = (
                "present bilateral" if "needle_decompression" in self._active_supports else "diminished bilateral"
            )
        elif updates.get("breath_sounds") in (None, ""):
            updates["breath_sounds"] = "present bilateral"

        if self._scenario is not None and self._scenario.scenario_id == "hemorrhagic_shock":
            if "control_bleeding" in self._active_supports:
                flow_rate = 25.0
            else:
                blood_volume = float(updates["blood_volume_ml"] or 4700.0)
                flow_rate = max(0.0, min(180.0, (5200.0 - blood_volume) * 0.4 + 60.0))
            updates["active_hemorrhages"] = {"right_leg": round(flow_rate, 1)} if flow_rate > 5.0 else {}
        else:
            updates["active_hemorrhages"] = {}

        if updates.get("systolic_bp_mmhg", 110.0) < 95.0 or updates.get("blood_volume_ml", 5500.0) < 5000.0:
            updates["lactate_trend"] = "worsening"
        elif self._active_supports & {"give_fluids", "control_bleeding", "give_oxygen", "needle_decompression"}:
            updates["lactate_trend"] = "improving"
        else:
            updates["lactate_trend"] = "stable"

        alerts: list[str] = []
        if updates["spo2"] < 0.92:
            alerts.append("hypoxemia")
        if updates["heart_rate_bpm"] > 110:
            alerts.append("tachycardia")
        if updates["systolic_bp_mmhg"] < 95:
            alerts.append("hypotension")
        if updates["blood_volume_ml"] is not None and updates["blood_volume_ml"] < 5000:
            alerts.append("blood_loss")
        if updates["respiration_rate_bpm"] >= 24:
            alerts.append("tachypnea")
        if updates.get("shock_index") is not None and updates["shock_index"] >= 0.9:
            alerts.append("shock_index_elevated")
        if updates["systolic_bp_mmhg"] < 70 or updates["spo2"] < 0.75:
            alerts.append("cardiovascular_collapse")

        updates["active_alerts"] = alerts
        updates["mental_status"] = self._derive_mental_status(updates["spo2"], updates["systolic_bp_mmhg"])
        updates["done"] = "cardiovascular_collapse" in alerts
        return PatientState(**updates)

    def _available_tools(self) -> list[str]:
        return list(KNOWN_TOOL_NAMES)

    def _suggest_airway_support_mode(self) -> str:
        assert self._state is not None

        if self._state.spo2 is not None and self._state.spo2 < 0.85:
            return "bag_valve_mask"
        if self._state.spo2 is not None and self._state.spo2 < 0.9:
            return "cpap"
        if self._state.mental_status in {"pain", "unresponsive"}:
            return "tracheal"
        if self._state.mental_status == "verbal":
            return "oropharyngeal"
        return "nasopharyngeal"

    def _derive_mental_status(self, spo2: float, systolic_bp_mmhg: float) -> str:
        if spo2 < 0.75 or systolic_bp_mmhg < 60:
            return "unresponsive"
        if spo2 < 0.82 or systolic_bp_mmhg < 70:
            return "pain"
        if spo2 < 0.88 or systolic_bp_mmhg < 85:
            return "verbal"
        return "alert"

    def _tool_message(self, tool_name: str) -> str:
        if tool_name == "give_oxygen":
            return "Supplemental oxygen started."
        if tool_name == "give_fluids":
            return "Fluid resuscitation initiated."
        if tool_name == "control_bleeding":
            return "Bleeding control measures applied."
        if tool_name == "position_patient":
            return "Patient repositioned for support."
        if tool_name == "airway_support":
            return "Airway support applied."
        if tool_name == "give_pressor":
            return "Vasopressor support initiated."
        if tool_name == "needle_decompression":
            return "Needle decompression performed."
        if tool_name == "pericardiocentesis":
            return "Pericardiocentesis performed."
        return f"{tool_name} executed."

    def _configure_runtime_effects(self, kwargs: dict[str, object]) -> None:
        observation_noise_level = float(kwargs.get("observation_noise_level", self._observation_noise.config.noise_level))
        raw_time_pressure_enabled = kwargs.get("time_pressure_enabled", self._time_pressure.config.enabled)
        if isinstance(raw_time_pressure_enabled, str):
            time_pressure_enabled = coerce_boolean_argument(raw_time_pressure_enabled)
        else:
            time_pressure_enabled = bool(raw_time_pressure_enabled)
        time_pressure_onset_s = float(kwargs.get("time_pressure_onset_s", self._time_pressure.config.onset_s))
        time_pressure_escalation_per_minute = float(
            kwargs.get(
                "time_pressure_escalation_per_minute",
                self._time_pressure.config.escalation_per_minute,
            )
        )
        self._observation_noise = NoisyObservation(observation_noise_level)
        self._time_pressure = TimePressureMechanic(
            enabled=time_pressure_enabled,
            onset_s=time_pressure_onset_s,
            escalation_per_minute=time_pressure_escalation_per_minute,
        )

    def _build_observed_state(self) -> tuple[PatientState, dict[str, object]]:
        assert self._state is not None
        observed_state, noise_metadata = self._observation_noise.apply(
            self._state,
            rng=self._episode_observation_rng,
        )
        time_pressure_metadata = self._time_pressure.as_metadata(
            sim_time_s=self._state.sim_time_s,
            injury_severity=self._scenario.injury_severity if self._scenario is not None else 0.0,
            unstable=self._is_state_unstable(self._state),
        )
        return observed_state, {
            "observation_noise": noise_metadata,
            "time_pressure": time_pressure_metadata,
        }

    def _current_time_pressure_multiplier(self, state: PatientState) -> float:
        assert self._scenario is not None
        return self._time_pressure.deterioration_multiplier(
            sim_time_s=state.sim_time_s,
            injury_severity=self._scenario.injury_severity,
            unstable=self._is_state_unstable(state),
        )

    @staticmethod
    def _is_state_unstable(state: PatientState) -> bool:
        systolic = state.systolic_bp_mmhg if state.systolic_bp_mmhg is not None else 120.0
        spo2 = state.spo2 if state.spo2 is not None else 1.0
        return bool(state.active_alerts) or systolic < 95.0 or spo2 < 0.92 or state.mental_status != "alert"
