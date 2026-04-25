"""Pulse Physiology Env Environment Client."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import PulsePhysiologyAction, PulsePhysiologyObservation


class PulsePhysiologyEnv(
    EnvClient[PulsePhysiologyAction, PulsePhysiologyObservation, State]
):
    """Client for the Pulse-backed physiology environment."""

    def _step_payload(self, action: PulsePhysiologyAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[PulsePhysiologyObservation]:
        obs_data = payload.get("observation", {})
        observation = PulsePhysiologyObservation.model_validate(
            {
                **obs_data,
                "done": payload.get("done", obs_data.get("done", False)),
                "reward": payload.get("reward"),
            }
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
