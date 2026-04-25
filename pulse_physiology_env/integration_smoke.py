"""Consumer-side integration smoke checks for mock or real adapters.

Examples:

    python -m pulse_physiology_env.integration_smoke
    python -m pulse_physiology_env.integration_smoke --backend-class pulse_physiology_env.server.adapters:MockPulseAdapter
"""

from __future__ import annotations

import argparse
import importlib
import inspect
from pathlib import Path

from pulse_physiology_env.models import (
    PulsePhysiologyObservation,
    ToolAction,
    ToolResult,
)
from pulse_physiology_env.patient_state import PatientState
from pulse_physiology_env.real_backend import RealPulseBackend
from pulse_physiology_env.server.mock_scenarios import DEFAULT_MOCK_SCENARIO_ID
from pulse_physiology_env.tool_catalog import KNOWN_TOOL_NAMES
from pulse_physiology_env.tool_parser import parse_tool_action


REQUIRED_OBSERVATION_FIELDS = {
    "scenario_id",
    "patient_id",
    "sim_time_s",
    "heart_rate_bpm",
    "systolic_bp_mmhg",
    "diastolic_bp_mmhg",
    "spo2",
    "respiration_rate_bpm",
    "blood_volume_ml",
    "mental_status",
    "active_alerts",
    "done",
}

REQUIRED_ENVELOPE_FIELDS = {"observation", "reward", "done", "metadata", "tool_result", "error"}


class _ConstructorNoScenarioBackend:
    """Regression-only backend stub with a constructor that accepts no scenario args."""

    def __init__(self) -> None:
        self.default_scenario_id = DEFAULT_MOCK_SCENARIO_ID

    def set_default_scenario(self, scenario_id: str) -> None:
        """Set the backend's default scenario after construction."""

        self.default_scenario_id = scenario_id

    def reset(self, scenario_id: str | None = None):
        """Proxy reset calls into a fresh mock backend instance."""

        from pulse_physiology_env.server.adapters import MockPulseAdapter

        backend = MockPulseAdapter(default_scenario_id=scenario_id or self.default_scenario_id)
        return backend.reset(scenario_id or self.default_scenario_id)

    def step(self, action):
        """Proxy one step through a fresh mock backend instance."""

        from pulse_physiology_env.server.adapters import MockPulseAdapter

        backend = MockPulseAdapter(default_scenario_id=self.default_scenario_id)
        backend.reset(self.default_scenario_id)
        return backend.step(action)


class _FakeRealEnvironment:
    """Regression-only OpenEnv-like environment used to test the real wrapper shape."""

    def __init__(self) -> None:
        self._state = PatientState(
            scenario_id="baseline_stable",
            patient_id="real_wrapper_fixture",
            heart_rate_bpm=72.0,
            systolic_bp_mmhg=118.0,
            diastolic_bp_mmhg=76.0,
            mean_arterial_pressure_mmhg=90.0,
            spo2=0.98,
            respiration_rate_bpm=14.0,
            blood_volume_ml=5500.0,
        )

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs):
        """Return a deterministic observation shaped like the real runtime."""

        del seed, episode_id
        scenario_id = str(kwargs.get("scenario_id") or self._state.scenario_id)
        self._state = self._state.model_copy(update={"scenario_id": scenario_id, "sim_time_s": 0.0})
        return self._build_observation(
            reward=0.0,
            tool_result=None,
            error=None,
            step_count=0,
        )

    def step(self, action: ToolAction, timeout_s: float | None = None, **kwargs):
        """Advance the fake environment by one action and return an observation."""

        del timeout_s, kwargs
        self._state = self._state.model_copy(update={"sim_time_s": self._state.sim_time_s + 30.0})
        return self._build_observation(
            reward=0.5,
            tool_result=ToolResult(
                tool_name=action.tool_name,
                success=True,
                message="Fake real environment step succeeded.",
                state_changed=action.tool_name == "advance_time",
                changed_fields=["sim_time_s"] if action.tool_name == "advance_time" else [],
            ),
            error=None,
            step_count=1,
        )

    def close(self) -> None:
        """Match the optional close hook used by the real environment."""

        return None

    def _build_observation(
        self,
        *,
        reward: float,
        tool_result,
        error,
        step_count: int,
    ) -> PulsePhysiologyObservation:
        """Build a real-style observation payload for wrapper regression checks."""

        raw_tools = list(KNOWN_TOOL_NAMES) + ["perform_cpr", "apply_nasal_cannula"]
        metadata = {
            "step_count": step_count,
            "available_tools": raw_tools,
            "scenario_description": "Fake real environment for wrapper regression.",
        }
        return PulsePhysiologyObservation.from_patient_state(
            self._state,
            reward=reward,
            available_tools=raw_tools,
            tool_result=tool_result,
            error=error,
            metadata=metadata,
        )


def _set_scenario_if_supported(backend, scenario_id: str) -> None:
    """Set the default scenario when a backend exposes a dedicated mutator hook.

    The mock and real adapters may temporarily diverge in constructor shape
    during integration, so the smoke test cannot assume every backend accepts
    ``default_scenario_id`` at construction time.
    """

    for method_name in ("set_default_scenario_id", "set_default_scenario", "set_scenario"):
        method = getattr(backend, method_name, None)
        if callable(method):
            method(scenario_id)
            return


def _load_backend(backend_class_path: str, scenario_id: str):
    """Instantiate the requested backend while tolerating constructor differences."""

    module_name, class_name = backend_class_path.split(":", 1)
    module = importlib.import_module(module_name)
    backend_cls = getattr(module, class_name)
    init_signature = inspect.signature(backend_cls)
    init_parameters = init_signature.parameters

    if "default_scenario_id" in init_parameters:
        return backend_cls(default_scenario_id=scenario_id)

    backend = backend_cls()
    _set_scenario_if_supported(backend, scenario_id)
    return backend


def _assert(condition: bool, message: str) -> None:
    """Abort the smoke run with a concise message when an assertion fails."""

    if not condition:
        raise SystemExit(message)


def _check_response_shape(result, label: str) -> None:
    """Validate that a backend response matches the consumer-side envelope contract."""

    payload = result.model_dump()
    _assert(REQUIRED_ENVELOPE_FIELDS <= set(payload), f"{label}: missing top-level response keys")
    observation = payload["observation"]
    _assert(REQUIRED_OBSERVATION_FIELDS <= set(observation), f"{label}: missing required observation fields")
    _assert(payload["done"] == observation["done"], f"{label}: done mismatch between envelope and observation")
    _assert(isinstance(payload["metadata"].get("available_tools"), list), f"{label}: metadata.available_tools must be a list")


def _regression_check_constructor_flexibility() -> None:
    """Ensure backend loading works even when the constructor omits scenario kwargs."""

    backend = _load_backend(
        "pulse_physiology_env.integration_smoke:_ConstructorNoScenarioBackend",
        DEFAULT_MOCK_SCENARIO_ID,
    )
    _assert(
        getattr(backend, "default_scenario_id", None) == DEFAULT_MOCK_SCENARIO_ID,
        "constructor_flexibility: scenario setter fallback did not prime the backend",
    )


def _regression_check_real_backend_wrapper() -> None:
    """Ensure the real-backend wrapper converts raw observations into response envelopes."""

    backend = RealPulseBackend(
        default_scenario_id=DEFAULT_MOCK_SCENARIO_ID,
        environment_factory=_FakeRealEnvironment,
    )
    reset_result = backend.reset(DEFAULT_MOCK_SCENARIO_ID)
    _check_response_shape(reset_result, "real_wrapper_reset")
    _assert(
        reset_result.observation.available_tools == list(KNOWN_TOOL_NAMES),
        "real_wrapper_reset: available_tools should be filtered to the frozen consumer tool set",
    )
    _assert(
        set(reset_result.observation.metadata.get("raw_available_tools", [])) >= {"perform_cpr", "apply_nasal_cannula"},
        "real_wrapper_reset: raw runtime tools should remain visible in metadata",
    )
    step_result = backend.step(ToolAction(tool_name="get_vitals", arguments={}))
    _check_response_shape(step_result, "real_wrapper_step")
    _assert(step_result.tool_result is not None, "real_wrapper_step: tool_result should be preserved")


def _regression_check_readme_frontmatter() -> None:
    """Ensure README frontmatter uses a readable emoji entry without mojibake."""

    readme_path = Path(__file__).resolve().parent / "README.md"
    frontmatter = readme_path.read_text(encoding="utf-8").splitlines()[:5]
    _assert(
        any(line.startswith("emoji: ") for line in frontmatter),
        "readme_frontmatter: expected an emoji frontmatter entry in README frontmatter",
    )
    _assert(
        not any("ðŸ" in line or "Ã°Å¸" in line for line in frontmatter),
        "readme_frontmatter: found mojibake in README frontmatter",
    )


def _regression_check_argument_normalization() -> None:
    """Ensure harmless formatting noise is normalized instead of penalized."""

    oxygen_action = parse_tool_action(
        '{"tool_name":"Give Oxygen","arguments":{"flow_lpm":"2LPM"}}'
    )
    _assert(oxygen_action.tool_name == "give_oxygen", "argument_normalization: tool_name should canonicalize")
    _assert(
        oxygen_action.arguments["flow_lpm"] == 2.0,
        "argument_normalization: oxygen flow should coerce to float",
    )

    bleed_action = parse_tool_action(
        '{"tool_name":"control_bleeding","arguments":{"method":"Tourniquet"}}'
    )
    _assert(
        bleed_action.arguments["method"] == "tourniquet",
        "argument_normalization: choice values should canonicalize",
    )

    pressor_action = parse_tool_action(
        '{"tool_name":"give_pressor","arguments":{"stop":"false"}}'
    )
    _assert(
        pressor_action.arguments["stop"] is False,
        "argument_normalization: boolean-like strings should coerce to booleans",
    )


def main() -> None:
    """Run lightweight contract checks against a mock or real backend implementation."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend-class",
        default="pulse_physiology_env.server.adapters:MockPulseAdapter",
        help="Module path to the adapter class in module:Class format.",
    )
    parser.add_argument("--scenario", default=DEFAULT_MOCK_SCENARIO_ID)
    args = parser.parse_args()

    backend = _load_backend(args.backend_class, args.scenario)

    print("Integration smoke check\n")
    print(f"backend_class: {args.backend_class}")
    print(f"scenario: {args.scenario}\n")

    _regression_check_constructor_flexibility()
    print("PASS constructor flexibility")
    _regression_check_real_backend_wrapper()
    print("PASS real backend wrapper shape")
    _regression_check_readme_frontmatter()
    print("PASS README frontmatter encoding")
    _regression_check_argument_normalization()
    print("PASS argument normalization")

    reset_result = backend.reset(args.scenario)
    _check_response_shape(reset_result, "reset")
    print("PASS reset envelope")

    valid_action = ToolAction(tool_name="get_vitals", arguments={})
    valid_result = backend.step(valid_action)
    _check_response_shape(valid_result, "valid_step")
    _assert(valid_result.tool_result is not None, "valid_step: tool_result must be present")
    print("PASS valid step envelope")

    invalid_tool_result = backend.step(ToolAction(tool_name="not_a_real_tool", arguments={}))
    _check_response_shape(invalid_tool_result, "invalid_tool")
    _assert(invalid_tool_result.error is not None, "invalid_tool: structured error expected")
    _assert(
        invalid_tool_result.error.code in {"UNKNOWN_TOOL", "UNSUPPORTED_TOOL"},
        "invalid_tool: expected UNKNOWN_TOOL or UNSUPPORTED_TOOL",
    )
    print("PASS unknown tool handling")

    invalid_arg_result = backend.step(ToolAction(tool_name="advance_time", arguments={"seconds": -5}))
    _check_response_shape(invalid_arg_result, "invalid_argument")
    _assert(invalid_arg_result.error is not None, "invalid_argument: structured error expected")
    _assert(invalid_arg_result.error.code == "INVALID_ARGUMENT", "invalid_argument: expected INVALID_ARGUMENT")
    print("PASS invalid argument handling")

    available_tools = valid_result.metadata.available_tools
    _assert(
        set(available_tools).issubset(set(KNOWN_TOOL_NAMES)),
        "available tools must stay within the known consumer tool set",
    )
    print("PASS available tool contract")

    print("\nIntegration smoke passed.")


if __name__ == "__main__":
    main()
