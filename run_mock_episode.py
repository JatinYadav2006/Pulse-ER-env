"""Run one policy against either the mock or real Pulse-ER backend.

Examples:

    python -m pulse_physiology_env.run_mock_episode --backend mock --scenario respiratory_distress --policy expert
    python -m pulse_physiology_env.run_mock_episode --backend real --scenario polytrauma_demo --policy expert
"""

from __future__ import annotations

import argparse

from pulse_physiology_env.episode_runner import EpisodeRunner
from pulse_physiology_env.policies import RandomPolicy, build_expert_policy, build_no_action_policy
from pulse_physiology_env.real_backend import RealPulseBackend
from pulse_physiology_env.server.adapters import MockPulseAdapter
from pulse_physiology_env.server.mock_scenarios import DEFAULT_MOCK_SCENARIO_ID, MOCK_SCENARIOS
from pulse_physiology_env.trajectory_io import append_trace_jsonl, write_trace_json


DEFAULT_REAL_SCENARIO_ID = "polytrauma_demo"
DEFAULT_REAL_MAX_STEPS = 12
DEFAULT_MOCK_MAX_STEPS = 8


def make_policy(policy_name: str, seed: int):
    """Create a supported policy instance."""

    if policy_name == "expert":
        return build_expert_policy()
    if policy_name == "random":
        return RandomPolicy(seed=seed)
    if policy_name == "no_action":
        return build_no_action_policy()
    raise ValueError(f"Unsupported policy '{policy_name}'")


def make_backend(
    backend_name: str,
    scenario: str,
    *,
    observation_noise_level: float = 0.0,
    time_pressure_enabled: bool = False,
    time_pressure_onset_s: float = 180.0,
    time_pressure_escalation_per_minute: float = 0.15,
):
    """Create the requested backend while keeping mock as the safe default."""

    if backend_name == "mock":
        return MockPulseAdapter(
            default_scenario_id=scenario,
            observation_noise_level=observation_noise_level,
            time_pressure_enabled=time_pressure_enabled,
            time_pressure_onset_s=time_pressure_onset_s,
            time_pressure_escalation_per_minute=time_pressure_escalation_per_minute,
        )
    if backend_name == "real":
        return RealPulseBackend(
            default_scenario_id=scenario,
            observation_noise_level=observation_noise_level,
            time_pressure_enabled=time_pressure_enabled,
            time_pressure_onset_s=time_pressure_onset_s,
            time_pressure_escalation_per_minute=time_pressure_escalation_per_minute,
        )
    raise ValueError(f"Unsupported backend '{backend_name}'")


def resolve_scenario(backend_name: str, requested_scenario: str | None) -> str:
    """Choose a sensible default scenario for the selected backend."""

    if requested_scenario:
        return requested_scenario
    if backend_name == "real":
        return DEFAULT_REAL_SCENARIO_ID
    return DEFAULT_MOCK_SCENARIO_ID


def validate_scenario(backend_name: str, scenario: str) -> None:
    """Fail early on invalid mock scenario IDs while letting the real backend own its IDs."""

    if backend_name == "mock" and scenario not in MOCK_SCENARIOS:
        valid = ", ".join(sorted(MOCK_SCENARIOS))
        raise ValueError(f"Unknown mock scenario '{scenario}'. Expected one of: {valid}")


def resolve_max_steps(backend_name: str, requested_max_steps: int | None) -> int:
    """Choose a backend-appropriate step horizon when the caller does not provide one."""

    if requested_max_steps is not None:
        return requested_max_steps
    if backend_name == "real":
        return DEFAULT_REAL_MAX_STEPS
    return DEFAULT_MOCK_MAX_STEPS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="mock", choices=("mock", "real"))
    parser.add_argument("--scenario")
    parser.add_argument("--policy", default="expert", choices=("expert", "random", "no_action"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--observation-noise-level", type=float, default=0.0)
    parser.add_argument("--time-pressure", action="store_true")
    parser.add_argument("--time-pressure-onset-s", type=float, default=180.0)
    parser.add_argument("--time-pressure-escalation-per-minute", type=float, default=0.15)
    parser.add_argument("--trace-json")
    parser.add_argument("--trace-jsonl")
    args = parser.parse_args()

    scenario = resolve_scenario(args.backend, args.scenario)
    validate_scenario(args.backend, scenario)
    max_steps = resolve_max_steps(args.backend, args.max_steps)

    policy = make_policy(args.policy, args.seed)
    backend = make_backend(
        args.backend,
        scenario,
        observation_noise_level=args.observation_noise_level,
        time_pressure_enabled=args.time_pressure,
        time_pressure_onset_s=args.time_pressure_onset_s,
        time_pressure_escalation_per_minute=args.time_pressure_escalation_per_minute,
    )
    runner = EpisodeRunner(backend=backend, max_steps=max_steps)

    try:
        trace = runner.run(policy=policy, scenario_id=scenario)
    finally:
        close_method = getattr(backend, "close", None)
        if callable(close_method):
            close_method()

    print("Episode summary")
    for key, value in trace.summary().items():
        print(f"  {key}: {value}")

    print("\nAction trace")
    for step in trace.steps:
        print(
            f"  step={step.step_index}"
            f" tool={step.action.tool_name}"
            f" reward={step.reward:.3f}"
            f" done={step.done}"
        )
        if step.tool_result is not None:
            print(f"    result={step.tool_result['message']}")
        if step.error is not None:
            print(f"    error={step.error['code']}: {step.error['message']}")

    if trace.events:
        print("\nRunner events")
        for event in trace.events:
            print(f"  - {event}")

    if args.trace_json:
        write_trace_json(trace, args.trace_json)
        print(f"\nWrote JSON trace to {args.trace_json}")
    if args.trace_jsonl:
        append_trace_jsonl(trace, args.trace_jsonl)
        print(f"\nAppended JSONL trace to {args.trace_jsonl}")


if __name__ == "__main__":
    main()
