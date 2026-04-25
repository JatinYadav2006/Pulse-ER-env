"""Consumer-side wrapper that adapts the real OpenEnv runtime to PatientBackend.

The real backend returns ``PulsePhysiologyObservation`` objects directly, while
Person 2's runner and validation stack expect ``EnvironmentResponse`` envelopes.
This thin wrapper preserves Person 1's runtime behavior and reshapes it for the
consumer-side pipeline without touching engine internals.
"""

from __future__ import annotations

from typing import Callable

from .models import EnvironmentResponse, ObservationMetadata, PulsePhysiologyObservation, ToolAction
from .patient_state import PatientState
from .server.adapters import PatientBackend


_REAL_SCENARIO_ALIASES = {
    "baseline_stable": "polytrauma_demo",
    "respiratory_distress": "trauma_easy_soldier",
    "hemorrhagic_shock": "trauma_hard_underweight",
}


class RealPulseBackend(PatientBackend):
    """Adapt ``PulsePhysiologyEnvironment`` to the ``PatientBackend`` interface."""

    def __init__(
        self,
        default_scenario_id: str | None = None,
        *,
        environment_factory: Callable[[], object] | None = None,
    ) -> None:
        self._default_scenario_id = default_scenario_id
        self._environment = self._build_environment(environment_factory)
        self._latest_observation: PulsePhysiologyObservation | None = None

    def reset(self, scenario_id: str | None = None) -> EnvironmentResponse:
        """Reset the real runtime and wrap its observation in the consumer envelope."""

        selected_scenario_id = scenario_id or self._default_scenario_id
        if selected_scenario_id is None:
            observation = self._environment.reset()
        else:
            observation = self._environment.reset(
                scenario_id=self._resolve_real_scenario_id(selected_scenario_id)
            )
        return self._wrap_observation(observation)

    def step(self, action: ToolAction) -> EnvironmentResponse:
        """Execute one action against the real runtime and wrap the response."""

        observation = self._environment.step(action)
        return self._wrap_observation(observation)

    def get_state(self) -> PatientState:
        """Reconstruct a ``PatientState`` view from the latest wrapped observation."""

        if self._latest_observation is None:
            raise RuntimeError("RealPulseBackend has not been reset yet.")

        payload = self._latest_observation.model_dump()
        state_payload = {
            field_name: payload[field_name]
            for field_name in PatientState.model_fields
            if field_name in payload
        }
        return PatientState(**state_payload)

    def close(self) -> None:
        """Close the underlying environment when it exposes a close hook."""

        close_method = getattr(self._environment, "close", None)
        if callable(close_method):
            close_method()

    @staticmethod
    def _resolve_real_scenario_id(scenario_id: str) -> str:
        """Map consumer-side mock aliases onto the nearest real runtime scenarios.

        The Person 2 pipeline historically used mock scenario names such as
        ``baseline_stable``. The real runtime exposes a different scenario ID
        set, so the wrapper translates only these known aliases and otherwise
        passes the provided value through untouched.
        """

        return _REAL_SCENARIO_ALIASES.get(scenario_id, scenario_id)

    @staticmethod
    def _build_environment(environment_factory: Callable[[], object] | None) -> object:
        """Instantiate the real environment lazily to avoid import-time runtime coupling."""

        if environment_factory is not None:
            return environment_factory()

        try:
            from .server.pulse_physiology_env_environment import PulsePhysiologyEnvironment
        except Exception as exc:  # pragma: no cover - depends on local Pulse/OpenEnv runtime
            raise RuntimeError(
                "Could not import PulsePhysiologyEnvironment. The real backend currently requires "
                "the Pulse/OpenEnv runtime stack and a working Python 3.12 installation."
            ) from exc
        return PulsePhysiologyEnvironment()

    def _wrap_observation(self, observation: PulsePhysiologyObservation) -> EnvironmentResponse:
        """Normalize a real observation into the standard ``EnvironmentResponse`` envelope."""

        if not isinstance(observation, PulsePhysiologyObservation):
            observation = PulsePhysiologyObservation.model_validate(observation)

        metadata_dict = dict(observation.metadata or {})
        available_tools = list(
            observation.available_tools
            or metadata_dict.get("available_tools")
            or []
        )
        metadata = ObservationMetadata(
            step_count=int(metadata_dict.get("step_count", 0)),
            available_tools=available_tools,
        )
        wrapped_observation = observation.model_copy(
            update={
                "available_tools": available_tools,
                "metadata": metadata_dict,
            }
        )
        self._latest_observation = wrapped_observation
        return EnvironmentResponse(
            observation=wrapped_observation,
            reward=float(wrapped_observation.reward or 0.0),
            done=wrapped_observation.done,
            metadata=metadata,
            tool_result=wrapped_observation.tool_result,
            error=wrapped_observation.error,
        )
