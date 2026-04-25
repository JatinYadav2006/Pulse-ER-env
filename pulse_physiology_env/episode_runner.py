"""Episode runner and trajectory logging for Pulse-ER policies."""

from __future__ import annotations

from dataclasses import dataclass

from .models import EnvironmentResponse, PulsePhysiologyObservation, ToolAction
from .policies import Policy
from .server.adapters import PatientBackend


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
            "sim_time_s": self.final_observation.sim_time_s,
            "heart_rate_bpm": self.final_observation.heart_rate_bpm,
            "systolic_bp_mmhg": self.final_observation.systolic_bp_mmhg,
            "diastolic_bp_mmhg": self.final_observation.diastolic_bp_mmhg,
            "spo2": self.final_observation.spo2,
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

    def run(self, policy: Policy, scenario_id: str) -> EpisodeTrace:
        """Execute one episode and capture its trajectory."""

        reset_result = self.backend.reset(scenario_id)
        policy.reset(scenario_id)

        current_observation = reset_result.observation
        total_reward = reset_result.reward
        steps: list[EpisodeStep] = []

        for step_index in range(self.max_steps):
            if current_observation.done:
                break

            action = policy.select_action(current_observation)
            result = self.backend.step(action)
            total_reward += result.reward
            observe_outcome = getattr(policy, "observe_outcome", None)
            if callable(observe_outcome):
                observe_outcome(action, result)

            steps.append(self._to_step(step_index, action, result))
            current_observation = result.observation

            if result.done or result.error is not None:
                break

        return EpisodeTrace(
            scenario_id=scenario_id,
            policy_name=policy.name,
            initial_observation=reset_result.observation,
            steps=tuple(steps),
            total_reward=round(total_reward, 3),
            final_observation=current_observation,
        )

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
