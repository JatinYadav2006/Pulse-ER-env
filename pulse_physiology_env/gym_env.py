"""Gym-style environment wrapper for online RL over Pulse-ER backends.

This wrapper keeps Person 1's runtime untouched and exposes a stable,
training-friendly interface on the Person 2 side. It uses discrete macro
actions backed by observation-aware default arguments so PPO/GRPO-style
training can start from a simple action space without losing clinically useful
parameterization.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .episode_runner import EpisodeTerminationReason
from .models import EnvironmentResponse, PulsePhysiologyObservation, ToolAction
from .real_backend import RealPulseBackend
from .server.adapters import MockPulseAdapter
from .server.mock_scenarios import DEFAULT_MOCK_SCENARIO_ID, MOCK_SCENARIOS
from .tool_availability import validate_tool_availability
from .tool_catalog import EXTENDED_TOOL_NAMES

try:  # pragma: no cover - optional convenience dependency
    from gymnasium import Env, spaces
except ImportError:  # pragma: no cover - keeps the wrapper usable without gymnasium installed
    class Env:
        """Very small fallback Env base used when gymnasium is unavailable."""

    @dataclass(frozen=True)
    class _Discrete:
        """Fallback discrete action space with a Gym-like surface."""

        n: int

        def sample(self) -> int:
            return random.randrange(self.n)

    @dataclass(frozen=True)
    class _Box:
        """Fallback box space exposing only the metadata the wrapper needs."""

        low: float
        high: float
        shape: tuple[int, ...]
        dtype: type = float

    class _Spaces:
        """Namespace shim so call sites can keep using ``spaces.X``."""

        Discrete = _Discrete
        Box = _Box

    spaces = _Spaces()


DEFAULT_REAL_SCENARIO_ID = "polytrauma_demo"
DEFAULT_REAL_MAX_EPISODE_STEPS = 12
DEFAULT_MOCK_MAX_EPISODE_STEPS = 8
INVALID_ACTION_PENALTY = -0.25

MENTAL_STATUS_ORDER = ("alert", "verbal", "pain", "unresponsive")
ALERT_FEATURES = (
    "hypoxemia",
    "tachycardia",
    "hypotension",
    "tachypnea",
    "blood_loss",
    "active_hemorrhage",
    "shock_index_elevated",
    "cardiac_arrest",
)


def _safe_divide(value: float | None, scale: float, default: float = 0.0) -> float:
    """Normalize a value safely when observations may omit fields."""

    if value is None:
        return default
    return float(value) / scale


def _clip_feature(value: float) -> float:
    """Bound a scalar feature to a stable range for simple online RL."""

    return max(-1.0, min(1.0, value))


def _map_value(observation: PulsePhysiologyObservation) -> float | None:
    """Return mean arterial pressure from explicit or derived blood pressure."""

    if observation.mean_arterial_pressure_mmhg is not None:
        return float(observation.mean_arterial_pressure_mmhg)
    if (
        observation.systolic_bp_mmhg is None
        or observation.diastolic_bp_mmhg is None
    ):
        return None
    return (
        float(observation.systolic_bp_mmhg)
        + 2.0 * float(observation.diastolic_bp_mmhg)
    ) / 3.0


def _highest_flow_hemorrhage(observation: PulsePhysiologyObservation) -> str | None:
    """Return the most severe active hemorrhage site when one exists."""

    if not observation.active_hemorrhages:
        return None
    return max(
        observation.active_hemorrhages.items(),
        key=lambda item: float(item[1] or 0.0),
    )[0]


def _bleeding_method(site: str | None) -> str:
    """Choose a reasonable default hemorrhage-control technique for a site."""

    normalized = (site or "").lower()
    if any(keyword in normalized for keyword in ("arm", "leg", "extremity")):
        return "tourniquet"
    if any(keyword in normalized for keyword in ("groin", "axilla", "junction")):
        return "hemostatic_dressing"
    return "pressure"


def _suggest_needle_side(observation: PulsePhysiologyObservation) -> str:
    """Infer the likely decompression side from bedside breath-sound text."""

    breath_sounds = (observation.breath_sounds or "").lower()
    if "left" in breath_sounds and any(token in breath_sounds for token in ("decreased", "absent")):
        return "left"
    if "right" in breath_sounds and any(token in breath_sounds for token in ("decreased", "absent")):
        return "right"
    return "left"


def _supports_oxygenation(observation: PulsePhysiologyObservation) -> bool:
    """Return whether the patient currently looks oxygenation-limited."""

    spo2 = observation.spo2 if observation.spo2 is not None else 1.0
    return spo2 < 0.94 or "hypoxemia" in set(observation.active_alerts or [])


def _action_from_tool_name(
    tool_name: str,
    observation: PulsePhysiologyObservation,
) -> ToolAction:
    """Translate a discrete macro action into a tool call with safe defaults."""

    if tool_name == "advance_time":
        return ToolAction(tool_name=tool_name, arguments={"seconds": 30.0})
    if tool_name == "give_oxygen":
        needs_more_oxygen = _supports_oxygenation(observation)
        return ToolAction(
            tool_name=tool_name,
            arguments={
                "device": "non_rebreather_mask" if needs_more_oxygen else "nasal_cannula",
                "flow_lpm": 15.0 if needs_more_oxygen else 4.0,
                "monitor_seconds": 60.0,
            },
        )
    if tool_name == "give_fluids":
        fluid_type = "blood" if observation.active_hemorrhages else "saline"
        return ToolAction(
            tool_name=tool_name,
            arguments={
                "fluid_type": fluid_type,
                "volume_ml": 500.0,
                "rate_ml_per_min": 150.0,
                "monitor_seconds": 60.0,
            },
        )
    if tool_name == "control_bleeding":
        site = _highest_flow_hemorrhage(observation)
        arguments: dict[str, Any] = {
            "method": _bleeding_method(site),
            "monitor_seconds": 60.0,
        }
        if site:
            arguments["site"] = site
        return ToolAction(tool_name=tool_name, arguments=arguments)
    if tool_name == "position_patient":
        map_value = _map_value(observation) or 0.0
        position = "supine" if map_value < 65.0 else "upright"
        return ToolAction(tool_name=tool_name, arguments={"position": position})
    if tool_name == "airway_support":
        support_type = "bag_valve_mask" if observation.mental_status == "unresponsive" else "cpap"
        return ToolAction(
            tool_name=tool_name,
            arguments={"support_type": support_type, "monitor_seconds": 60.0},
        )
    if tool_name == "give_pressor":
        return ToolAction(
            tool_name=tool_name,
            arguments={
                "pressor": "norepinephrine",
                "rate_ml_per_min": 7.5,
                "monitor_seconds": 60.0,
            },
        )
    if tool_name == "needle_decompression":
        return ToolAction(
            tool_name=tool_name,
            arguments={"side": _suggest_needle_side(observation), "monitor_seconds": 90.0},
        )
    if tool_name == "pericardiocentesis":
        return ToolAction(tool_name=tool_name, arguments={"monitor_seconds": 90.0})
    return ToolAction(tool_name=tool_name, arguments={})


def featurize_observation(observation: PulsePhysiologyObservation) -> list[float]:
    """Convert a clinical observation into a stable fixed-length feature vector."""

    map_value = _map_value(observation)
    alerts = set(observation.active_alerts or [])
    hemorrhage_total = sum(float(value or 0.0) for value in observation.active_hemorrhages.values())
    mental_status = getattr(observation.mental_status, "value", observation.mental_status)

    features = [
        _clip_feature(_safe_divide(observation.heart_rate_bpm, 180.0)),
        _clip_feature(_safe_divide(observation.systolic_bp_mmhg, 120.0)),
        _clip_feature(_safe_divide(observation.diastolic_bp_mmhg, 80.0)),
        _clip_feature(_safe_divide(map_value, 90.0)),
        _clip_feature(float(observation.spo2 or 0.0)),
        _clip_feature(_safe_divide(observation.respiration_rate_bpm, 30.0)),
        _clip_feature(_safe_divide(observation.blood_volume_ml, 5000.0)),
        _clip_feature(_safe_divide(observation.shock_index, 1.5)),
        _clip_feature(_safe_divide(observation.sim_time_s, 600.0)),
        _clip_feature(_safe_divide(hemorrhage_total, 800.0)),
        _clip_feature(_safe_divide(float(len(observation.active_hemorrhages)), 4.0)),
        _clip_feature(_safe_divide(float(len(alerts)), 8.0)),
        _clip_feature(_safe_divide(float(len(observation.active_infusions)), 4.0)),
        _clip_feature(_safe_divide(float(len(observation.ready_diagnostics)), 4.0)),
    ]

    for status_name in MENTAL_STATUS_ORDER:
        features.append(1.0 if mental_status == status_name else 0.0)
    for alert_name in ALERT_FEATURES:
        features.append(1.0 if alert_name in alerts else 0.0)

    return features


class PulseGymEnv(Env):
    """Gym-style RL environment over either the mock or real Pulse-ER backend."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        backend_name: str = "mock",
        scenario_id: str | None = None,
        max_episode_steps: int | None = None,
        invalid_action_penalty: float = INVALID_ACTION_PENALTY,
        seed: int = 0,
        observation_noise_level: float = 0.0,
        time_pressure_enabled: bool = False,
        time_pressure_onset_s: float = 180.0,
        time_pressure_escalation_per_minute: float = 0.15,
    ) -> None:
        self.backend_name = backend_name
        self.scenario_id = self._resolve_scenario(backend_name, scenario_id)
        self.max_episode_steps = self._resolve_max_episode_steps(backend_name, max_episode_steps)
        self.invalid_action_penalty = invalid_action_penalty
        self._rng = random.Random(seed)
        self._backend_kwargs = {
            "observation_noise_level": observation_noise_level,
            "time_pressure_enabled": time_pressure_enabled,
            "time_pressure_onset_s": time_pressure_onset_s,
            "time_pressure_escalation_per_minute": time_pressure_escalation_per_minute,
        }

        self._validate_scenario()
        self.backend = self._make_backend()
        self.tool_names = list(EXTENDED_TOOL_NAMES)
        self.action_space = spaces.Discrete(len(self.tool_names))
        feature_dim = len(featurize_observation(PulsePhysiologyObservation()))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(feature_dim,), dtype=float)

        self._step_count = 0
        self._current_response: EnvironmentResponse | None = None
        self._current_observation: PulsePhysiologyObservation | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[list[float], dict[str, Any]]:
        """Reset the backend and return an RL-friendly feature vector plus info."""

        if seed is not None:
            self._rng.seed(seed)

        reset_options = dict(options or {})
        requested_scenario = reset_options.pop("scenario_id", self.scenario_id)
        response = self.backend.reset(requested_scenario, **reset_options)
        self._step_count = 0
        self._current_response = response
        self._current_observation = response.observation
        return featurize_observation(response.observation), self._build_info(response, tool_name=None)

    def step(self, action_index: int) -> tuple[list[float], float, bool, bool, dict[str, Any]]:
        """Execute one discrete macro action and return Gym-style step outputs."""

        if self._current_observation is None or self._current_response is None:
            raise RuntimeError("PulseGymEnv.reset() must be called before step().")

        tool_name = self.tool_names[int(action_index)]
        available_tools = validate_tool_availability(self._current_observation.available_tools)
        self._step_count += 1

        if tool_name not in available_tools:
            terminated = self._is_terminal(self._current_observation)
            truncated = not terminated and self._step_count >= self.max_episode_steps
            info = self._build_invalid_action_info(tool_name)
            info["termination_reason"] = self._termination_reason(terminated, truncated)
            return (
                featurize_observation(self._current_observation),
                float(self.invalid_action_penalty),
                terminated,
                truncated,
                info,
            )

        action = _action_from_tool_name(tool_name, self._current_observation)
        response = self.backend.step(action)
        self._current_response = response
        self._current_observation = response.observation

        terminated = self._is_terminal(response.observation)
        truncated = not terminated and self._step_count >= self.max_episode_steps
        info = self._build_info(response, tool_name=tool_name, tool_action=action)
        info["termination_reason"] = self._termination_reason(terminated, truncated)
        return (
            featurize_observation(response.observation),
            float(response.reward),
            terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        """Close the underlying backend if it exposes a close hook."""

        close_method = getattr(self.backend, "close", None)
        if callable(close_method):
            close_method()

    def action_mask(self) -> list[bool]:
        """Return a boolean mask aligned with the discrete action space."""

        if self._current_observation is None:
            return [False] * len(self.tool_names)
        available = set(validate_tool_availability(self._current_observation.available_tools))
        return [tool_name in available for tool_name in self.tool_names]

    def _build_info(
        self,
        response: EnvironmentResponse,
        *,
        tool_name: str | None,
        tool_action: ToolAction | None = None,
    ) -> dict[str, Any]:
        """Build a rich RL info payload for logging, masking, and debugging."""

        return {
            "scenario_id": response.observation.scenario_id,
            "step_count": self._step_count,
            "tool_name": tool_name,
            "tool_action": tool_action.model_dump() if tool_action is not None else None,
            "tool_result": response.tool_result.model_dump() if response.tool_result else None,
            "error": response.error.model_dump() if response.error else None,
            "available_tools": list(response.observation.available_tools),
            "action_mask": self.action_mask(),
            "invalid_action": False,
            "observation": response.observation,
        }

    def _build_invalid_action_info(self, tool_name: str) -> dict[str, Any]:
        """Return a deterministic info payload for masked-out invalid actions."""

        return {
            "scenario_id": self._current_observation.scenario_id if self._current_observation else self.scenario_id,
            "step_count": self._step_count,
            "tool_name": tool_name,
            "tool_action": None,
            "tool_result": None,
            "error": {
                "code": "INVALID_ACTION_MASK",
                "message": f"Tool '{tool_name}' is not currently available.",
                "retryable": False,
            },
            "available_tools": list(self._current_observation.available_tools) if self._current_observation else [],
            "action_mask": self.action_mask(),
            "invalid_action": True,
            "observation": self._current_observation,
        }

    @staticmethod
    def _is_terminal(observation: PulsePhysiologyObservation) -> bool:
        """Detect terminal physiology for RL episode termination."""

        return observation.done or "cardiac_arrest" in set(observation.active_alerts or [])

    @staticmethod
    def _termination_reason(terminated: bool, truncated: bool) -> str | None:
        """Map Gym termination booleans onto the shared runner reason strings."""

        if terminated:
            return EpisodeTerminationReason.PATIENT_DEATH.value
        if truncated:
            return EpisodeTerminationReason.MAX_TIMESTEPS.value
        return None

    def _make_backend(self):
        """Instantiate the requested backend while keeping mock as the default-safe path."""

        if self.backend_name == "mock":
            return MockPulseAdapter(default_scenario_id=self.scenario_id, **self._backend_kwargs)
        if self.backend_name == "real":
            return RealPulseBackend(default_scenario_id=self.scenario_id, **self._backend_kwargs)
        raise ValueError(f"Unsupported backend '{self.backend_name}'.")

    @staticmethod
    def _resolve_scenario(backend_name: str, requested_scenario: str | None) -> str:
        """Choose a default scenario that matches the selected backend."""

        if requested_scenario:
            return requested_scenario
        if backend_name == "real":
            return DEFAULT_REAL_SCENARIO_ID
        return DEFAULT_MOCK_SCENARIO_ID

    @staticmethod
    def _resolve_max_episode_steps(backend_name: str, requested_steps: int | None) -> int:
        """Choose a backend-appropriate episode horizon."""

        if requested_steps is not None:
            return requested_steps
        if backend_name == "real":
            return DEFAULT_REAL_MAX_EPISODE_STEPS
        return DEFAULT_MOCK_MAX_EPISODE_STEPS

    def _validate_scenario(self) -> None:
        """Fail early on invalid mock scenarios while letting the real backend own its IDs."""

        if self.backend_name == "mock" and self.scenario_id not in MOCK_SCENARIOS:
            valid = ", ".join(sorted(MOCK_SCENARIOS))
            raise ValueError(f"Unknown mock scenario '{self.scenario_id}'. Expected one of: {valid}")
