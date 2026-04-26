"""Generate a minimal "before" graph using random tool calls on the mock backend."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pulse_physiology_env.episode_runner import EpisodeRunner
from pulse_physiology_env.policies import RandomPolicy
from pulse_physiology_env.server.adapters import MockPulseAdapter
from pulse_physiology_env.server.mock_scenarios import MOCK_SCENARIOS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="respiratory_distress", choices=sorted(MOCK_SCENARIOS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric", choices=("spo2", "reward"), default="spo2")
    parser.add_argument("--output", default="artifacts/random_before_plot.png")
    args = parser.parse_args()

    backend = MockPulseAdapter(default_scenario_id=args.scenario, seed=args.seed)
    runner = EpisodeRunner(backend=backend, max_steps=8)
    try:
        trace = runner.run(policy=RandomPolicy(seed=args.seed), scenario_id=args.scenario)
    finally:
        close_method = getattr(backend, "close", None)
        if callable(close_method):
            close_method()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.metric == "spo2":
        x_values = [0] + [step.step_index + 1 for step in trace.steps]
        y_values = [trace.initial_observation.spo2 * 100] + [step.observation.spo2 * 100 for step in trace.steps]
        y_label = "SpO2 (%)"
        color = "#d68c2f"
    else:
        x_values = [step.step_index + 1 for step in trace.steps]
        y_values = [step.reward for step in trace.steps]
        y_label = "Reward"
        color = "#cc4b5a"

    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor="white")
    ax.plot(x_values, y_values, marker="o", linewidth=2.4, color=color)
    ax.set_xlabel("Episode step")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.22)

    for x_value, y_value in zip(x_values, y_values, strict=True):
        label = f"{y_value:.1f}" if args.metric == "spo2" else f"{y_value:+.3f}"
        ax.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(output_path.resolve())


if __name__ == "__main__":
    main()
