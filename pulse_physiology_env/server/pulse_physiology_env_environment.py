"""Pulse-backed OpenEnv environment implementation."""

from __future__ import annotations

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

try:
    from ..models import PulsePhysiologyAction, PulsePhysiologyObservation
    from ..patient_state import PatientState
    from .pulse_engine_adapter import PulseEngineAdapter
    from .reward_engine import RewardBreakdown, RewardEngine, RewardTracker
    from .scenarios import DEFAULT_SCENARIO_ID, PatientProfile, ScenarioDefinition, get_scenario_definition
    from .tools import PulseToolExecutor
except ImportError:
    from models import PulsePhysiologyAction, PulsePhysiologyObservation
    from patient_state import PatientState
    from server.pulse_engine_adapter import PulseEngineAdapter
    from server.reward_engine import RewardBreakdown, RewardEngine, RewardTracker
    from server.scenarios import DEFAULT_SCENARIO_ID, PatientProfile, ScenarioDefinition, get_scenario_definition
    from server.tools import PulseToolExecutor


class PulsePhysiologyEnvironment(Environment):
    """A Pulse-backed tool environment for trauma and resuscitation workflows."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._adapter = PulseEngineAdapter()
        self._tool_executor = PulseToolExecutor(self._adapter)
        self._reward_engine = RewardEngine()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario: ScenarioDefinition = get_scenario_definition(DEFAULT_SCENARIO_ID)
        self._selected_patient: PatientProfile | None = None
        self._latest_patient_state: PatientState | None = None
        self._reward_tracker: RewardTracker | None = None
        self._last_reward_breakdown = RewardBreakdown()

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: object,
    ) -> PulsePhysiologyObservation:
        """Reset the environment and initialize the requested Pulse scenario."""

        scenario_id = kwargs.get("scenario_id")
        self._scenario = get_scenario_definition(str(scenario_id) if scenario_id is not None else DEFAULT_SCENARIO_ID)
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        rng = random.Random(seed)
        self._selected_patient = self._scenario.choose_patient(rng)

        patient_state = self._adapter.load_patient(
            state_file=self._selected_patient.state_file,
            scenario_id=self._scenario.scenario_id,
            scenario_difficulty=self._scenario.difficulty,
            patient_id=self._selected_patient.patient_id,
        )
        if self._scenario.setup is not None:
            self._scenario.setup(self._adapter)
            patient_state = self._adapter.get_full_state()

        patient_state = self._apply_episode_rules(patient_state)
        self._latest_patient_state = patient_state
        self._reward_tracker = self._reward_engine.start_episode(self._scenario, patient_state)
        self._last_reward_breakdown = RewardBreakdown(
            reward_profile=self._scenario.reward_profile,
            difficulty_multiplier=self._reward_engine.DIFFICULTY_MULTIPLIER[self._scenario.difficulty],
            action_budget_remaining=self._reward_tracker.action_budget_remaining,
        )
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
        if self._reward_tracker is None:
            self._reward_tracker = self._reward_engine.start_episode(self._scenario, previous_state)
        breakdown = self._reward_engine.score_step(
            self._reward_tracker,
            scenario=self._scenario,
            before=previous_state,
            after=current_state,
            action=action,
            success=execution.tool_result.success,
            had_error=execution.error is not None,
        )
        reward = breakdown.total
        self._latest_patient_state = current_state
        self._last_reward_breakdown = breakdown

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
            version="0.4.0",
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
            "scenario_difficulty": self._scenario.difficulty,
            "reward_profile": self._scenario.reward_profile,
            "patient_pool_size": len(self._scenario.patient_pool),
            "selected_state_file": self._selected_patient.state_file if self._selected_patient is not None else None,
            "action_budget_remaining": self._reward_tracker.action_budget_remaining if self._reward_tracker is not None else None,
            "reward_breakdown": self._last_reward_breakdown.as_metadata(),
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
