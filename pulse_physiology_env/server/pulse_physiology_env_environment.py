"""Pulse-backed OpenEnv environment implementation."""

from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

try:
    from ..models import PulsePhysiologyAction, PulsePhysiologyObservation
    from ..patient_state import PatientState
    from .pulse_engine_adapter import PulseEngineAdapter
    from .scenarios import DEFAULT_SCENARIO_ID, ScenarioDefinition, get_scenario_definition
    from .tools import PulseToolExecutor
except ImportError:
    from models import PulsePhysiologyAction, PulsePhysiologyObservation
    from patient_state import PatientState
    from server.pulse_engine_adapter import PulseEngineAdapter
    from server.scenarios import DEFAULT_SCENARIO_ID, ScenarioDefinition, get_scenario_definition
    from server.tools import PulseToolExecutor


class PulsePhysiologyEnvironment(Environment):
    """A Pulse-backed tool environment for trauma and resuscitation workflows."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._adapter = PulseEngineAdapter()
        self._tool_executor = PulseToolExecutor(self._adapter)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario: ScenarioDefinition = get_scenario_definition(DEFAULT_SCENARIO_ID)
        self._latest_patient_state: PatientState | None = None

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: object,
    ) -> PulsePhysiologyObservation:
        """Reset the environment and initialize the requested Pulse scenario."""

        del seed
        scenario_id = kwargs.get("scenario_id")
        self._scenario = get_scenario_definition(str(scenario_id) if scenario_id is not None else DEFAULT_SCENARIO_ID)
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)

        patient_state = self._adapter.load_patient(
            state_file=self._scenario.state_file,
            scenario_id=self._scenario.scenario_id,
            patient_id=self._scenario.patient_id,
        )
        if self._scenario.setup is not None:
            self._scenario.setup(self._adapter)
            patient_state = self._adapter.get_full_state()

        patient_state = self._apply_episode_rules(patient_state)
        self._latest_patient_state = patient_state
        return self._build_observation(
            patient_state,
            reward=0.0,
            tool_result=None,
            error=None,
        )

    def step(
        self,
        action: PulsePhysiologyAction,
        timeout_s: float | None = None,
        **kwargs: object,
    ) -> PulsePhysiologyObservation:
        """Execute a named tool against the current Pulse scenario."""

        del timeout_s, kwargs
        self._state.step_count += 1

        previous_state = self._latest_patient_state or self._apply_episode_rules(self._adapter.get_full_state())
        execution = self._tool_executor.execute(action)
        current_state = self._apply_episode_rules(execution.state)
        reward = self._compute_reward(previous_state, current_state, execution.tool_result.success, execution.error is not None)
        self._latest_patient_state = current_state

        return self._build_observation(
            current_state,
            reward=reward,
            tool_result=execution.tool_result,
            error=execution.error,
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""

        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        description = (
            "Pulse-backed trauma environment with engine-correct tools for airway, breathing, "
            "circulation, diagnostics, and decision support."
        )
        return EnvironmentMetadata(
            name="PulsePhysiologyEnvironment",
            description=description,
            version="0.2.0",
            author="OpenAI Codex",
        )

    def close(self) -> None:
        self._adapter.close()

    def _apply_episode_rules(self, state: PatientState) -> PatientState:
        done = state.done or state.sim_time_s >= self._scenario.max_time_s
        alerts = list(state.active_alerts)
        if state.sim_time_s >= self._scenario.max_time_s and "time_limit_reached" not in alerts:
            alerts.append("time_limit_reached")
        return state.model_copy(update={"done": done, "active_alerts": alerts})

    def _build_observation(
        self,
        state: PatientState,
        *,
        reward: float,
        tool_result,
        error,
    ) -> PulsePhysiologyObservation:
        metadata = {
            "step_count": self._state.step_count,
            "scenario_description": self._scenario.description,
            "available_tools": self._tool_executor.available_tools,
        }
        return PulsePhysiologyObservation.from_patient_state(
            state,
            reward=reward,
            available_tools=self._tool_executor.available_tools,
            tool_result=tool_result,
            error=error,
            metadata=metadata,
        )

    def _compute_reward(
        self,
        before: PatientState,
        after: PatientState,
        success: bool,
        had_error: bool,
    ) -> float:
        reward = -0.05

        reward += 8.0 * (self._score_spo2(after.spo2) - self._score_spo2(before.spo2))
        reward += 6.0 * (
            self._score_pressure(after.mean_arterial_pressure_mmhg)
            - self._score_pressure(before.mean_arterial_pressure_mmhg)
        )
        reward += 5.0 * (
            self._score_shock_index(before.shock_index) - self._score_shock_index(after.shock_index)
        )
        reward += 2.0 * (
            self._score_mental_status(after.mental_status) - self._score_mental_status(before.mental_status)
        )

        if before.blood_volume_ml is not None and after.blood_volume_ml is not None:
            reward += (after.blood_volume_ml - before.blood_volume_ml) / 500.0

        reward += 0.3 * max(0, len(before.active_alerts) - len(after.active_alerts))
        reward -= 0.3 * max(0, len(after.active_alerts) - len(before.active_alerts))

        if after.done and not before.done:
            reward -= 25.0
        if had_error:
            reward -= 0.75
        elif not success:
            reward -= 0.25

        return float(max(-30.0, min(30.0, reward)))

    @staticmethod
    def _score_spo2(value: float | None) -> float:
        if value is None:
            return 0.0
        return max(0.0, min((value - 0.75) / 0.2, 1.0))

    @staticmethod
    def _score_pressure(value: float | None) -> float:
        if value is None:
            return 0.0
        if value < 40:
            return 0.0
        if value < 65:
            return (value - 40) / 25.0 * 0.6
        if value <= 90:
            return 0.6 + ((value - 65) / 25.0) * 0.4
        return max(0.0, 1.0 - min((value - 90) / 40.0, 1.0) * 0.2)

    @staticmethod
    def _score_shock_index(value: float | None) -> float:
        if value is None:
            return 0.0
        if value <= 0.7:
            return 1.0
        if value >= 1.5:
            return 0.0
        return 1.0 - ((value - 0.7) / 0.8)

    @staticmethod
    def _score_mental_status(value: str) -> float:
        return {
            "alert": 1.0,
            "verbal": 0.66,
            "pain": 0.33,
            "unresponsive": 0.0,
        }.get(value, 0.0)
