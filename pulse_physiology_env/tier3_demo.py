"""Demo runner for Tier 3 workflows.

Examples:

    python -m pulse_physiology_env.tier3_demo --scenario hemorrhagic_shock
    python -m pulse_physiology_env.tier3_demo --scenario respiratory_distress --warmup-policy expert
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

from pulse_physiology_env.demo_llm_policy import heuristic_infer_fn
from pulse_physiology_env.episode_runner import EpisodeRunner
from pulse_physiology_env.policies import LLMPolicy, build_expert_policy
from pulse_physiology_env.server.adapters import MockPulseAdapter
from pulse_physiology_env.server.mock_scenarios import DEFAULT_MOCK_SCENARIO_ID, MOCK_SCENARIOS
from pulse_physiology_env.tier3_workflows import (
    build_episode_report,
    build_triage_summary,
    explain_deterioration,
    generate_intervention_plan,
    recommend_next_step,
)


def _warmup_trace(scenario_id: str, warmup_policy: str, max_steps: int):
    backend = MockPulseAdapter(default_scenario_id=scenario_id)
    runner = EpisodeRunner(backend=backend, max_steps=max_steps)
    if warmup_policy == "expert":
        policy = build_expert_policy()
        return runner.run(policy=policy, scenario_id=scenario_id)
    if warmup_policy == "llm_demo":
        policy = LLMPolicy(infer_fn=heuristic_infer_fn, name="llm_demo")
        return runner.run(policy=policy, scenario_id=scenario_id)
    reset_result = backend.reset(scenario_id)
    return SimpleNamespace(
        final_observation=reset_result.observation,
        initial_observation=reset_result.observation,
        steps=(),
        scenario_id=scenario_id,
        policy_name="none",
        total_reward=0.0,
        num_steps=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=DEFAULT_MOCK_SCENARIO_ID, choices=sorted(MOCK_SCENARIOS))
    parser.add_argument("--warmup-policy", default="none", choices=("none", "expert", "llm_demo"))
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    trace = _warmup_trace(args.scenario, args.warmup_policy, args.max_steps)
    observation = trace.final_observation
    previous_observation = trace.steps[-1].observation if trace.steps else None
    observation_history = [trace.initial_observation, *(step.observation for step in trace.steps)]

    triage = build_triage_summary(observation)
    recommendation = recommend_next_step(observation)
    explanation = explain_deterioration(
        observation,
        previous_observation,
        observations=observation_history,
    )
    plan = generate_intervention_plan(observation)
    episode_report = build_episode_report(trace) if trace.steps else None

    print("Tier 3 workflow demo\n")
    print("Triage summary")
    print(json.dumps(triage.model_dump(), indent=2))
    print("\nNext-step recommendation")
    print(json.dumps(recommendation.model_dump(), indent=2))
    print("\nDeterioration explanation")
    print(json.dumps(explanation.model_dump(), indent=2))
    print("\nIntervention plan")
    print(json.dumps(plan.model_dump(), indent=2))
    if episode_report is not None:
        print("\nEpisode report")
        print(json.dumps(episode_report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
