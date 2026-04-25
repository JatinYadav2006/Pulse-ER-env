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

from pulse_physiology_env.models import INITIAL_TOOL_NAMES, ToolAction
from pulse_physiology_env.server.mock_scenarios import DEFAULT_MOCK_SCENARIO_ID


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
        self.default_scenario_id = scenario_id

    def reset(self, scenario_id: str | None = None):
        from pulse_physiology_env.server.adapters import MockPulseAdapter

        backend = MockPulseAdapter(default_scenario_id=scenario_id or self.default_scenario_id)
        return backend.reset(scenario_id or self.default_scenario_id)

    def step(self, action):
        from pulse_physiology_env.server.adapters import MockPulseAdapter

        backend = MockPulseAdapter(default_scenario_id=self.default_scenario_id)
        backend.reset(self.default_scenario_id)
        return backend.step(action)


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
    if not condition:
        raise SystemExit(message)


def _check_response_shape(result, label: str) -> None:
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


def _regression_check_readme_frontmatter() -> None:
    """Ensure README frontmatter does not contain the old mojibake emoji string."""

    readme_path = Path(__file__).resolve().parent / "README.md"
    frontmatter = readme_path.read_text(encoding="utf-8").splitlines()[:5]
    _assert(
        any(line == "emoji: 🩺" for line in frontmatter),
        "readme_frontmatter: expected a clean UTF-8 emoji entry in README frontmatter",
    )
    _assert(
        not any("ðŸ" in line for line in frontmatter),
        "readme_frontmatter: found mojibake in README frontmatter",
    )


def main() -> None:
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
    _regression_check_readme_frontmatter()
    print("PASS README frontmatter encoding")

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
    _assert(invalid_tool_result.error.code == "UNKNOWN_TOOL", "invalid_tool: expected UNKNOWN_TOOL")
    print("PASS unknown tool handling")

    invalid_arg_result = backend.step(ToolAction(tool_name="advance_time", arguments={"seconds": -5}))
    _check_response_shape(invalid_arg_result, "invalid_argument")
    _assert(invalid_arg_result.error is not None, "invalid_argument: structured error expected")
    _assert(invalid_arg_result.error.code == "INVALID_ARGUMENT", "invalid_argument: expected INVALID_ARGUMENT")
    print("PASS invalid argument handling")

    available_tools = valid_result.metadata.available_tools
    _assert(set(available_tools).issubset(set(INITIAL_TOOL_NAMES)), "available tools must stay within the frozen tool set")
    print("PASS available tool contract")

    print("\nIntegration smoke passed.")


if __name__ == "__main__":
    main()
