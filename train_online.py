"""Minimal online RL smoke trainer for Pulse-ER.

This is intentionally lightweight: it proves that the current Pulse-ER stack
can support online RL over the new Gym-style wrapper without introducing a
heavy framework dependency. The wrapper is compatible with future PPO/GRPO
work, while this trainer gives us an immediate end-to-end learning harness.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

from .gym_env import PulseGymEnv


@dataclass(frozen=True)
class EpisodeMetrics:
    """Compact summary of one training or evaluation episode."""

    total_reward: float
    num_steps: int
    terminated: bool
    truncated: bool
    termination_reason: str | None


class LinearSoftmaxPolicy:
    """Tiny REINFORCE policy over handcrafted clinical features."""

    def __init__(self, num_actions: int, feature_dim: int, *, seed: int = 0) -> None:
        self.num_actions = num_actions
        self.feature_dim = feature_dim
        self.rng = random.Random(seed)
        self.weights = [
            [self.rng.uniform(-0.01, 0.01) for _ in range(feature_dim)]
            for _ in range(num_actions)
        ]
        self.bias = [0.0 for _ in range(num_actions)]

    def sample_action(self, observation: list[float], action_mask: list[bool]) -> tuple[int, list[float]]:
        """Sample a masked action from the current policy."""

        probs = self.action_probabilities(observation, action_mask)
        action_index = self.rng.choices(range(self.num_actions), weights=probs, k=1)[0]
        return int(action_index), probs

    def greedy_action(self, observation: list[float], action_mask: list[bool]) -> int:
        """Choose the highest-probability currently valid action."""

        probs = self.action_probabilities(observation, action_mask)
        return max(range(len(probs)), key=lambda index: probs[index])

    def action_probabilities(self, observation: list[float], action_mask: list[bool]) -> list[float]:
        """Return a masked softmax distribution over actions."""

        logits = [
            sum(weight * value for weight, value in zip(action_weights, observation, strict=True)) + bias_value
            for action_weights, bias_value in zip(self.weights, self.bias, strict=True)
        ]
        valid_logits = [logit for logit, is_valid in zip(logits, action_mask, strict=True) if is_valid]
        if not valid_logits:
            raise RuntimeError("No valid actions are available for the current observation.")
        max_logit = max(valid_logits)
        exp_logits = [
            math.exp(logit - max_logit) if is_valid else 0.0
            for logit, is_valid in zip(logits, action_mask, strict=True)
        ]
        total = sum(exp_logits)
        return [value / total for value in exp_logits]

    def update_episode(
        self,
        transitions: list[dict],
        *,
        learning_rate: float,
        gamma: float,
    ) -> None:
        """Apply one REINFORCE update from a completed episode."""

        if not transitions:
            return

        returns: list[float] = []
        running_return = 0.0
        for transition in reversed(transitions):
            running_return = float(transition["reward"]) + gamma * running_return
            returns.append(running_return)
        returns.reverse()

        baseline = sum(returns) / len(returns)
        advantages = [max(-10.0, min(10.0, value - baseline)) for value in returns]

        for transition, advantage in zip(transitions, advantages, strict=True):
            observation_vector = [float(value) for value in transition["observation"]]
            probs = [float(value) for value in transition["probs"]]
            chosen_index = int(transition["action_index"])
            for action_index in range(self.num_actions):
                grad_logit = (1.0 if action_index == chosen_index else 0.0) - probs[action_index]
                update_scale = learning_rate * float(advantage) * grad_logit
                self.bias[action_index] += update_scale
                for feature_index, feature_value in enumerate(observation_vector):
                    self.weights[action_index][feature_index] += update_scale * feature_value


def run_episode(
    env: PulseGymEnv,
    policy: LinearSoftmaxPolicy,
    *,
    train: bool,
    gamma: float,
    learning_rate: float,
    seed: int | None = None,
) -> EpisodeMetrics:
    """Run one episode and optionally update the policy online."""

    observation, info = env.reset(seed=seed)
    transitions: list[dict] = []
    total_reward = 0.0
    num_steps = 0
    terminated = False
    truncated = False
    termination_reason: str | None = None

    while True:
        action_mask = list(info["action_mask"])
        if train:
            action_index, probs = policy.sample_action(observation, action_mask)
        else:
            action_index = policy.greedy_action(observation, action_mask)
            probs = policy.action_probabilities(observation, action_mask)

        next_observation, reward, terminated, truncated, info = env.step(action_index)
        transitions.append(
            {
                "observation": observation,
                "action_index": action_index,
                "reward": reward,
                "probs": probs,
            }
        )
        total_reward += float(reward)
        num_steps += 1
        observation = next_observation
        termination_reason = info.get("termination_reason")
        if terminated or truncated:
            break

    if train:
        policy.update_episode(
            transitions,
            learning_rate=learning_rate,
            gamma=gamma,
        )

    return EpisodeMetrics(
        total_reward=round(total_reward, 3),
        num_steps=num_steps,
        terminated=terminated,
        truncated=truncated,
        termination_reason=termination_reason,
    )


def evaluate_policy(
    policy: LinearSoftmaxPolicy,
    *,
    backend_name: str,
    scenario_id: str | None,
    max_episode_steps: int,
    eval_episodes: int,
    seed: int,
) -> dict[str, float]:
    """Run deterministic evaluation episodes and summarize policy quality."""

    rewards: list[float] = []
    lengths: list[int] = []
    survival_count = 0

    env = PulseGymEnv(
        backend_name=backend_name,
        scenario_id=scenario_id,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )
    try:
        for episode_index in range(eval_episodes):
            metrics = run_episode(
                env,
                policy,
                train=False,
                gamma=1.0,
                learning_rate=0.0,
                seed=seed + episode_index,
            )
            rewards.append(metrics.total_reward)
            lengths.append(metrics.num_steps)
            if not metrics.terminated:
                survival_count += 1
    finally:
        env.close()

    return {
        "avg_reward": round(sum(rewards) / max(1, len(rewards)), 3),
        "avg_steps": round(sum(lengths) / max(1, len(lengths)), 2),
        "survival_rate": round(survival_count / max(1, len(rewards)), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="mock", choices=("mock", "real"))
    parser.add_argument("--scenario")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = PulseGymEnv(
        backend_name=args.backend,
        scenario_id=args.scenario,
        max_episode_steps=args.max_steps,
        seed=args.seed,
    )
    try:
        initial_observation, _ = env.reset(seed=args.seed)
        policy = LinearSoftmaxPolicy(
            num_actions=env.action_space.n,
            feature_dim=len(initial_observation),
            seed=args.seed,
        )
        initial_eval = evaluate_policy(
            policy,
            backend_name=args.backend,
            scenario_id=args.scenario,
            max_episode_steps=env.max_episode_steps,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
        )
        print("Initial evaluation")
        for key, value in initial_eval.items():
            print(f"  {key}: {value}")

        recent_rewards: list[float] = []
        for episode_index in range(1, args.episodes + 1):
            metrics = run_episode(
                env,
                policy,
                train=True,
                gamma=args.gamma,
                learning_rate=args.learning_rate,
                seed=args.seed + episode_index,
            )
            recent_rewards.append(metrics.total_reward)
            recent_rewards = recent_rewards[-10:]
            if episode_index == 1 or episode_index % max(1, math.ceil(args.episodes / 5)) == 0:
                rolling_reward = round(sum(recent_rewards) / len(recent_rewards), 3)
                print(
                    f"Episode {episode_index}/{args.episodes}: "
                    f"reward={metrics.total_reward:.3f} "
                    f"steps={metrics.num_steps} "
                    f"termination={metrics.termination_reason} "
                    f"rolling_avg_reward={rolling_reward:.3f}"
                )

        final_eval = evaluate_policy(
            policy,
            backend_name=args.backend,
            scenario_id=args.scenario,
            max_episode_steps=env.max_episode_steps,
            eval_episodes=args.eval_episodes,
            seed=args.seed + 1000,
        )
        print("\nFinal evaluation")
        for key, value in final_eval.items():
            print(f"  {key}: {value}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
