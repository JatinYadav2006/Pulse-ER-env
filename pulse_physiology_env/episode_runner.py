"""Episode runner and trajectory logging for Pulse-ER policies."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from .models import EnvironmentResponse, PulsePhysiologyObservation, ToolAction
from .policies import Policy
from .server.adapters import PatientBackend
from .tool_availability import ToolAvailabilityError, validate_tool_availability


class EpisodeTerminationReason(str, Enum):
    """Why an episode stopped from the consumer-side runner's perspective."""

    PATIENT_DEATH = "patient_death"
    MAX_TIMESTEPS = "max_timesteps"
    FATAL_BACKEND_ERROR = "fatal_backend_error"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class EpisodeStep:
    """One action/result pair in a trajectory."""

    step_index: int
    action: ToolAction
    reward: float
    done: bool
    observation: PulsePhysiologyObservation
    tool_result: dict | None
    error: dict | None


@dataclass(frozen=True)
class EpisodeTrace:
    """End-to-end record of one episode."""

    scenario_id: str
    policy_name: str
    initial_observation: PulsePhysiologyObservation
    steps: tuple[EpisodeStep, ...]
    total_reward: float
    final_observation: PulsePhysiologyObservation
    termination_reason: EpisodeTerminationReason
    action_budget_remaining: int | None = None
    events: tuple[str, ...] = ()

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def summary(self) -> dict:
        """Compact episode summary for CLI tools and logging."""

        mental_status = self.final_observation.mental_status
        mental_status_value = getattr(mental_status, "value", mental_status)

        return {
            "scenario_id": self.scenario_id,
            "policy_name": self.policy_name,
            "num_steps": self.num_steps,
            "total_reward": round(self.total_reward, 3),
            "done": self.final_observation.done,
            "termination_reason": self.termination_reason.value,
            "action_budget_remaining": self.action_budget_remaining,
            "sim_time_s": self.final_observation.sim_time_s,
            "heart_rate_bpm": self.final_observation.heart_rate_bpm,
            "systolic_bp_mmhg": self.final_observation.systolic_bp_mmhg,
            "diastolic_bp_mmhg": self.final_observation.diastolic_bp_mmhg,
            "spo2": self.final_observation.spo2,
            "spo2_percent": round(self.final_observation.spo2 * 100, 1)
            if self.final_observation.spo2 is not None
            else None,
            "respiration_rate_bpm": self.final_observation.respiration_rate_bpm,
            "blood_volume_ml": self.final_observation.blood_volume_ml,
            "mental_status": mental_status_value,
            "active_alerts": self.final_observation.active_alerts,
        }


@dataclass
class EpisodeRunner:
    """Reusable runner for executing policies against a backend."""

    backend: PatientBackend
    max_steps: int = 8
    action_budget: int | None = None
    max_retry_attempts: int = 3
    retry_backoff_s: float = 0.01

    @staticmethod
    def _is_terminal_observation(observation: PulsePhysiologyObservation) -> bool:
        """Detect terminal physiology from the observation even before `done` is set.

        The real Pulse runtime may surface cardiac-arrest state through alerts
        before a subsequent tool call flips `done=True`. Detecting that here
        avoids wasting retry attempts or extra advance-time calls in obviously
        terminal states.
        """

        if observation.done:
            return True
        active_alerts = set(observation.active_alerts or [])
        return "cardiac_arrest" in active_alerts

    def run(self, policy: Policy, scenario_id: str) -> EpisodeTrace:
        """Execute one episode and capture its trajectory."""

        reset_result = self.backend.reset(scenario_id)
        policy.reset(scenario_id)

        current_observation = reset_result.observation
        total_reward = reset_result.reward
        steps: list[EpisodeStep] = []
        events: list[str] = []
        remaining_action_budget = self.action_budget
        termination_reason = EpisodeTerminationReason.MAX_TIMESTEPS

        if self._is_terminal_observation(current_observation):
            termination_reason = EpisodeTerminationReason.PATIENT_DEATH

        for step_index in range(self.max_steps):
            if remaining_action_budget is not None and remaining_action_budget <= 0:
                events.append("Action budget exhausted before the next decision could be made.")
                termination_reason = EpisodeTerminationReason.BUDGET_EXHAUSTED
                break
            if self._is_terminal_observation(current_observation):
                if "cardiac_arrest" in set(current_observation.active_alerts or []):
                    events.append(
                        "Terminal physiology detected from observation state; ending episode without another backend step."
                    )
                termination_reason = EpisodeTerminationReason.PATIENT_DEATH
                break

            try:
                validate_tool_availability(current_observation.available_tools)
                action = policy.select_action(current_observation)
            except ToolAvailabilityError as exc:
                events.append(f"Fatal backend error: {exc}")
                termination_reason = EpisodeTerminationReason.FATAL_BACKEND_ERROR
                break

            result = self._step_with_retry(action, events)
            if remaining_action_budget is not None:
                remaining_action_budget -= 1
            total_reward += result.reward
            observe_outcome = getattr(policy, "observe_outcome", None)
            if callable(observe_outcome):
                observe_outcome(action, result)

            steps.append(self._to_step(step_index, action, result))
            current_observation = result.observation

            if self._is_terminal_observation(current_observation):
                if "cardiac_arrest" in set(current_observation.active_alerts or []):
                    events.append(
                        "Terminal physiology detected after step result; ending episode without further retries."
                    )
                termination_reason = EpisodeTerminationReason.PATIENT_DEATH
                break

            if result.error is not None:
                if result.error.retryable:
                    events.append(
                        f"Retryable error persisted for {action.tool_name}; action skipped after "
                        f"{self.max_retry_attempts} attempts."
                    )
                    continue
                events.append(
                    f"Fatal backend error from {action.tool_name}: {result.error.code} - {result.error.message}"
                )
                termination_reason = EpisodeTerminationReason.FATAL_BACKEND_ERROR
                break
        else:
            termination_reason = EpisodeTerminationReason.MAX_TIMESTEPS

        return EpisodeTrace(
            scenario_id=reset_result.observation.scenario_id,
            policy_name=policy.name,
            initial_observation=reset_result.observation,
            steps=tuple(steps),
            total_reward=round(total_reward, 3),
            final_observation=current_observation,
            termination_reason=termination_reason,
            action_budget_remaining=remaining_action_budget,
            events=tuple(events),
        )

    def _step_with_retry(
        self,
        action: ToolAction,
        events: list[str],
    ) -> EnvironmentResponse:
        """Execute one backend step with bounded retries for transient failures."""

        result: EnvironmentResponse | None = None
        for attempt_index in range(1, self.max_retry_attempts + 1):
            result = self.backend.step(action)
            if self._is_terminal_observation(result.observation):
                events.append(
                    f"Terminal physiology detected after {action.tool_name}; skipping further retries."
                )
                return result
            if result.error is None or not result.error.retryable:
                return result

            events.append(
                f"Retryable error on {action.tool_name} attempt {attempt_index}/{self.max_retry_attempts}: "
                f"{result.error.code} - {result.error.message}"
            )
            if attempt_index < self.max_retry_attempts and self.retry_backoff_s > 0:
                time.sleep(self.retry_backoff_s)

        assert result is not None
        return result

    def _to_step(
        self,
        step_index: int,
        action: ToolAction,
        result: EnvironmentResponse,
    ) -> EpisodeStep:
        return EpisodeStep(
            step_index=step_index,
            action=action.model_copy(deep=True),
            reward=result.reward,
            done=result.done,
            observation=result.observation,
            tool_result=result.tool_result.model_dump() if result.tool_result else None,
            error=result.error.model_dump() if result.error else None,
        )
