"""Generate reusable mock trajectory artifacts for training and evaluation.

Examples:

    python -m pulse_physiology_env.generate_seed_trajectories
    python -m pulse_physiology_env.generate_seed_trajectories --random-seeds 16
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from pulse_physiology_env.demo_llm_policy import heuristic_infer_fn
from pulse_physiology_env.episode_runner import EpisodeRunner, EpisodeTrace
from pulse_physiology_env.policies import (
    LLMPolicy,
    RandomPolicy,
    build_expert_policy,
    build_no_action_policy,
)
from pulse_physiology_env.server.adapters import MockPulseAdapter
from pulse_physiology_env.server.mock_scenarios import MOCK_SCENARIOS
from pulse_physiology_env.trajectory_io import append_trace_jsonl


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_JSONL = PACKAGE_DIR / "artifacts" / "seed_trajectories.jsonl"
DEFAULT_SUMMARY_JSON = PACKAGE_DIR / "artifacts" / "seed_trajectories_summary.json"


@dataclass(frozen=True)
class PolicyRun:
    """Configuration for one generated trajectory family."""

    policy_name: str
    seed: int | None = None


def _policy_runs(selected_policies: list[str], random_seeds: int) -> list[PolicyRun]:
    runs: list[PolicyRun] = []
    for policy_name in selected_policies:
        if policy_name == "random":
            runs.extend(PolicyRun(policy_name="random", seed=seed) for seed in range(random_seeds))
        else:
            runs.append(PolicyRun(policy_name=policy_name))
    return runs


def _build_policy(run: PolicyRun):
    if run.policy_name == "expert":
        return build_expert_policy()
    if run.policy_name == "no_action":
        return build_no_action_policy()
    if run.policy_name == "llm_demo":
        return LLMPolicy(infer_fn=heuristic_infer_fn, name="llm_demo")
    if run.policy_name == "random":
        return RandomPolicy(seed=run.seed or 0, name=f"random_seed_{run.seed or 0}")
    raise ValueError(f"Unsupported policy '{run.policy_name}'")


def _run_trace(scenario_id: str, run: PolicyRun, max_steps: int) -> EpisodeTrace:
    backend = MockPulseAdapter(default_scenario_id=scenario_id)
    runner = EpisodeRunner(backend=backend, max_steps=max_steps)
    policy = _build_policy(run)
    return runner.run(policy=policy, scenario_id=scenario_id)


def _policy_group(policy_name: str) -> str:
    return "random" if policy_name.startswith("random_seed_") else policy_name


def _summary_payload(traces: list[EpisodeTrace], output_jsonl: str) -> dict:
    grouped_rewards: dict[str, dict[str, list[float]]] = {}
    for trace in traces:
        policy_group = _policy_group(trace.policy_name)
        grouped_rewards.setdefault(policy_group, {}).setdefault(trace.scenario_id, []).append(trace.total_reward)

    summary = {
        "output_jsonl": output_jsonl,
        "num_traces": len(traces),
        "policies": sorted(grouped_rewards),
        "scenarios": sorted(MOCK_SCENARIOS),
        "average_reward_by_policy": {},
    }
    for policy_name, scenario_map in grouped_rewards.items():
        scenario_summary = {}
        for scenario_id, rewards in scenario_map.items():
            scenario_summary[scenario_id] = round(sum(rewards) / len(rewards), 3)
        summary["average_reward_by_policy"][policy_name] = scenario_summary

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-jsonl",
        default=DEFAULT_OUTPUT_JSONL,
        help="Path to the generated JSONL trajectory dataset.",
    )
    parser.add_argument(
        "--summary-json",
        default=DEFAULT_SUMMARY_JSON,
        help="Path to the summary JSON file.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["expert", "random", "no_action", "llm_demo"],
        choices=["expert", "random", "no_action", "llm_demo"],
        help="Policies to include in the generated artifact set.",
    )
    parser.add_argument(
        "--random-seeds",
        type=int,
        default=8,
        help="Number of random-policy seeds to generate when random is selected.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum episode length for each generated trace.",
    )
    args = parser.parse_args()

    output_jsonl = Path(args.output_jsonl)
    summary_json = Path(args.summary_json)
    if not output_jsonl.is_absolute():
        output_jsonl = (PACKAGE_DIR.parent / output_jsonl).resolve()
    if not summary_json.is_absolute():
        summary_json = (PACKAGE_DIR.parent / summary_json).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("", encoding="utf-8")

    traces: list[EpisodeTrace] = []
    for scenario_id in MOCK_SCENARIOS:
        for run in _policy_runs(args.policies, args.random_seeds):
            trace = _run_trace(scenario_id, run, args.max_steps)
            traces.append(trace)
            append_trace_jsonl(trace, output_jsonl)

    summary = _summary_payload(traces, str(output_jsonl))
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Generated seed trajectories")
    print(f"  traces: {len(traces)}")
    print(f"  jsonl: {output_jsonl}")
    print(f"  summary: {summary_json}")
    for policy_name, scenario_map in summary["average_reward_by_policy"].items():
        averages = ", ".join(f"{scenario_id}={reward:.3f}" for scenario_id, reward in scenario_map.items())
        print(f"  {policy_name}: {averages}")


if __name__ == "__main__":
    main()
