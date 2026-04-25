"""Evaluation harness for comparing mock policies.

Run from the repo root with:

    python -m pulse_physiology_env.eval_mock
"""

from __future__ import annotations

from dataclasses import dataclass

from pulse_physiology_env.demo_llm_policy import heuristic_infer_fn
from pulse_physiology_env.episode_runner import EpisodeRunner
from pulse_physiology_env.policies import (
    LLMPolicy,
    RandomPolicy,
    build_expert_policy,
    build_no_action_policy,
)
from pulse_physiology_env.server.adapters import MockPulseAdapter
from pulse_physiology_env.server.mock_scenarios import MOCK_SCENARIOS


@dataclass(frozen=True)
class PolicyScore:
    """Summary score for one policy over all scenarios."""

    policy_name: str
    per_scenario: dict[str, float]

    @property
    def average_reward(self) -> float:
        return round(sum(self.per_scenario.values()) / len(self.per_scenario), 3)


def score_policy(policy_factory, policy_name: str) -> PolicyScore:
    """Evaluate one policy factory across all mock scenarios."""

    per_scenario: dict[str, float] = {}
    for scenario_id in MOCK_SCENARIOS:
        backend = MockPulseAdapter(default_scenario_id=scenario_id)
        runner = EpisodeRunner(backend=backend, max_steps=8)
        policy = policy_factory(scenario_id)
        trace = runner.run(policy=policy, scenario_id=scenario_id)
        per_scenario[scenario_id] = trace.total_reward

    return PolicyScore(policy_name=policy_name, per_scenario=per_scenario)


def score_random_policy(num_seeds: int = 12) -> PolicyScore:
    """Evaluate the mean reward of seeded random policies."""

    per_scenario: dict[str, float] = {}
    for scenario_id in MOCK_SCENARIOS:
        rewards = []
        for seed in range(num_seeds):
            backend = MockPulseAdapter(default_scenario_id=scenario_id)
            runner = EpisodeRunner(backend=backend, max_steps=8)
            policy = RandomPolicy(seed=seed)
            trace = runner.run(policy=policy, scenario_id=scenario_id)
            rewards.append(trace.total_reward)
        per_scenario[scenario_id] = round(sum(rewards) / len(rewards), 3)

    return PolicyScore(policy_name="random", per_scenario=per_scenario)


def print_policy_score(score: PolicyScore) -> None:
    """Pretty-print one policy summary."""

    print(f"{score.policy_name} policy")
    for scenario_id, reward in score.per_scenario.items():
        print(f"  {scenario_id}: {reward:.3f}")
    print(f"  average: {score.average_reward:.3f}")


def main() -> None:
    """Compare expert, random, and no-action baselines."""

    expert = score_policy(lambda scenario_id: build_expert_policy(), "expert")
    llm_demo = score_policy(
        lambda scenario_id: LLMPolicy(infer_fn=heuristic_infer_fn, name="llm_demo"),
        "llm_demo",
    )
    random_policy = score_random_policy()
    no_action = score_policy(lambda scenario_id: build_no_action_policy(), "no_action")

    print("Mock policy evaluation\n")
    print_policy_score(expert)
    print()
    print_policy_score(llm_demo)
    print()
    print_policy_score(random_policy)
    print()
    print_policy_score(no_action)
    print()

    if not (
        expert.average_reward > llm_demo.average_reward > random_policy.average_reward > no_action.average_reward
    ):
        raise SystemExit(
            "Policy ranking check failed: expected expert > llm_demo > random > no_action on average."
        )

    print("PASS expert > llm_demo > random > no_action")


if __name__ == "__main__":
    main()
