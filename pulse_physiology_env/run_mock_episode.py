"""Run one policy against one mock scenario.

Examples:

    python -m pulse_physiology_env.run_mock_episode --scenario respiratory_distress --policy expert
    python -m pulse_physiology_env.run_mock_episode --scenario hemorrhagic_shock --policy random
"""

from __future__ import annotations

import argparse

from pulse_physiology_env.episode_runner import EpisodeRunner
from pulse_physiology_env.policies import RandomPolicy, build_expert_policy, build_no_action_policy
from pulse_physiology_env.server.adapters import MockPulseAdapter
from pulse_physiology_env.server.mock_scenarios import DEFAULT_MOCK_SCENARIO_ID, MOCK_SCENARIOS
from pulse_physiology_env.trajectory_io import append_trace_jsonl, write_trace_json


def make_policy(policy_name: str, seed: int):
    """Create a supported policy instance."""

    if policy_name == "expert":
        return build_expert_policy()
    if policy_name == "random":
        return RandomPolicy(seed=seed)
    if policy_name == "no_action":
        return build_no_action_policy()
    raise ValueError(f"Unsupported policy '{policy_name}'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=DEFAULT_MOCK_SCENARIO_ID, choices=sorted(MOCK_SCENARIOS))
    parser.add_argument("--policy", default="expert", choices=("expert", "random", "no_action"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--trace-json")
    parser.add_argument("--trace-jsonl")
    args = parser.parse_args()

    policy = make_policy(args.policy, args.seed)
    backend = MockPulseAdapter(default_scenario_id=args.scenario)
    runner = EpisodeRunner(backend=backend, max_steps=args.max_steps)
    trace = runner.run(policy=policy, scenario_id=args.scenario)

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

    if args.trace_json:
        write_trace_json(trace, args.trace_json)
        print(f"\nWrote JSON trace to {args.trace_json}")
    if args.trace_jsonl:
        append_trace_jsonl(trace, args.trace_jsonl)
        print(f"\nAppended JSONL trace to {args.trace_jsonl}")


if __name__ == "__main__":
    main()
