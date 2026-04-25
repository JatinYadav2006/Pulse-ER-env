"""Backend adapters for swapping mock and real Pulse runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod

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
from ..rewards import compute_reward
from ..tool_catalog import ToolValidationError, validate_tool_arguments
from .mock_scenarios import DEFAULT_MOCK_SCENARIO_ID, MOCK_SCENARIOS, MockScenarioDefinition


class PatientBackend(ABC):
    """Stable interface between Person 2's stack and the backend runtime."""

    @abstractmethod
    def reset(self, scenario_id: str | None = None) -> EnvironmentResponse:
        """Reset the environment and return the initial response."""

    @abstractmethod
    def step(self, action: ToolAction) -> EnvironmentResponse:
        """Apply one action and return the next response."""

    @abstractmethod
    def get_state(self) -> PatientState:
        """Return the latest patient state."""


class MockPulseAdapter(PatientBackend):
    """Deterministic backend used by Person 2 before real Pulse integration exists."""

    def __init__(self, default_scenario_id: str = DEFAULT_MOCK_SCENARIO_ID):
        self._default_scenario_id = default_scenario_id
        self._scenario: MockScenarioDefinition | None = None
        self._state: PatientState | None = None
        self._step_count = 0
        self._active_supports: set[str] = set()

    def reset(self, scenario_id: str | None = None) -> EnvironmentResponse:
        selected_scenario_id = scenario_id or self._default_scenario_id
        scenario = MOCK_SCENARIOS[selected_scenario_id]

        self._scenario = scenario
        self._state = self._refresh_state(scenario.initial_state.model_copy(deep=True))
        self._step_count = 0
        self._active_supports = set()

        return self._build_response(
            reward=0.0,
            tool_result=ToolResult(
                tool_name="reset",
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

        if action.tool_name not in INITIAL_TOOL_NAMES:
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
                allowed_tools=INITIAL_TOOL_NAMES,
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
        elif action.tool_name in {"summarize_state", "check_deterioration", "recommend_next_step", "get_vitals"}:
            result = self._read_only_tool(action.tool_name)
        else:
            result = self._apply_intervention(action)

        if result.error is not None:
            return result

        self._state = self._refresh_state(self._state)
        reward = compute_reward(
            previous_state,
            self._state,
            action.tool_name,
            self._scenario.recommended_actions,
        ).total
        changed_fields = self._changed_fields(previous_state, self._state)

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

        for field_name, delta in self._scenario.deterioration_per_30s.items():
            adjusted_delta = self._deterioration_delta(field_name, delta)
            current_value = updates.get(field_name)
            if current_value is None:
                continue
            updates[field_name] = current_value + adjusted_delta * scale

        updates["sim_time_s"] = self._state.sim_time_s + seconds
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

        if action.tool_name not in self._scenario.tool_effects:
            return self._error_response(
                code="UNSUPPORTED_IN_SCENARIO",
                message=f"{action.tool_name} is not modeled for scenario '{self._scenario.scenario_id}'.",
                retryable=False,
                tool_name=action.tool_name,
            )

        updates = self._state.model_dump()
        effect_scale = self._intervention_scale(action.tool_name)
        for field_name, delta in self._scenario.tool_effects[action.tool_name].items():
            current_value = updates.get(field_name)
            if current_value is None:
                continue
            updates[field_name] = current_value + (delta * effect_scale)

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

        return scale

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

        return EnvironmentResponse(
            observation=PulsePhysiologyObservation.from_patient_state(
                self._state,
                reward=reward,
                available_tools=available_tools,
                tool_result=tool_result,
                error=error,
                metadata={"step_count": self._step_count},
            ),
            reward=reward,
            done=self._state.done,
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

        return EnvironmentResponse(
            observation=PulsePhysiologyObservation.from_patient_state(
                state,
                reward=-1.0,
                available_tools=available_tools,
                error=ToolError(code=code, message=message, retryable=retryable),
                metadata={"step_count": self._step_count},
            ),
            reward=-1.0,
            done=state.done,
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
        if updates["systolic_bp_mmhg"] < 70 or updates["spo2"] < 0.75:
            alerts.append("cardiovascular_collapse")

        updates["active_alerts"] = alerts
        updates["mental_status"] = self._derive_mental_status(updates["spo2"], updates["systolic_bp_mmhg"])
        updates["done"] = "cardiovascular_collapse" in alerts
        return PatientState(**updates)

    def _available_tools(self) -> list[str]:
        if self._scenario is None:
            return list(INITIAL_TOOL_NAMES)
        scenario_tools = set(self._scenario.tool_effects)
        return [tool_name for tool_name in INITIAL_TOOL_NAMES if tool_name in scenario_tools]

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
        return f"{tool_name} executed."
