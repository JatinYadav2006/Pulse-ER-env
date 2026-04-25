"""Preview the policy prompt for a mock scenario state.

Examples:

    python -m pulse_physiology_env.preview_prompt --scenario respiratory_distress
    python -m pulse_physiology_env.preview_prompt --scenario hemorrhagic_shock --warmup-policy expert --warmup-steps 2
"""

from __future__ import annotations

import argparse

from pulse_physiology_env.policies import RandomPolicy, build_expert_policy, build_no_action_policy
from pulse_physiology_env.prompt_builder import build_policy_prompt
from pulse_physiology_env.server.adapters import MockPulseAdapter
from pulse_physiology_env.server.mock_scenarios import DEFAULT_MOCK_SCENARIO_ID, MOCK_SCENARIOS


def make_policy(policy_name: str, seed: int):
    """Create a supported warmup policy."""

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
    parser.add_argument("--warmup-policy", choices=("expert", "random", "no_action"))
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    adapter = MockPulseAdapter(default_scenario_id=args.scenario)
    result = adapter.reset(args.scenario)
    observation = result.observation

    if args.warmup_policy and args.warmup_steps > 0:
        policy = make_policy(args.warmup_policy, args.seed)
        policy.reset(args.scenario)
        for _ in range(args.warmup_steps):
            if observation.done:
                break
            tool_action = policy.select_action(observation)
            result = adapter.step(tool_action)
            observation = result.observation

    prompt = build_policy_prompt(observation)
    print(prompt)


if __name__ == "__main__":
    main()
