"""Consumer-side integration smoke checks for mock or real adapters.

Examples:

    python -m pulse_physiology_env.integration_smoke
    python -m pulse_physiology_env.integration_smoke --backend-class pulse_physiology_env.server.adapters:MockPulseAdapter
"""

from __future__ import annotations

import argparse
import importlib

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


def _load_backend(backend_class_path: str, scenario_id: str):
    module_name, class_name = backend_class_path.split(":", 1)
    module = importlib.import_module(module_name)
    backend_cls = getattr(module, class_name)
    return backend_cls(default_scenario_id=scenario_id)


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
