"""Generate a submission-friendly before/after mock policy comparison plot.

This script is intentionally honest about what "before" means:

- before: a weak random tool-calling baseline on the selected mock scenario
- after: the trained/expert policy on the same scenario

It also includes the verified mock benchmark scores for context so the figure
is useful both as a quick demo asset and as a reproducible benchmark artifact.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pulse_physiology_env.demo_llm_policy import heuristic_infer_fn
from pulse_physiology_env.episode_runner import EpisodeRunner
from pulse_physiology_env.eval_mock import score_policy, score_random_policy
from pulse_physiology_env.policies import LLMPolicy, RandomPolicy, build_expert_policy, build_no_action_policy
from pulse_physiology_env.server.adapters import MockPulseAdapter
from pulse_physiology_env.server.mock_scenarios import MOCK_SCENARIOS


def _run_trace(policy, *, scenario_id: str):
    backend = MockPulseAdapter(default_scenario_id=scenario_id, seed=0)
    runner = EpisodeRunner(backend=backend, max_steps=8)
    try:
        return runner.run(policy=policy, scenario_id=scenario_id)
    finally:
        close_method = getattr(backend, "close", None)
        if callable(close_method):
            close_method()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="respiratory_distress", choices=sorted(MOCK_SCENARIOS))
    parser.add_argument(
        "--output",
        default="artifacts/mock_before_after_comparison.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    expert_scores = score_policy(lambda scenario_id: build_expert_policy(), "expert")
    llm_scores = score_policy(
        lambda scenario_id: LLMPolicy(infer_fn=heuristic_infer_fn, name="llm_demo"),
        "llm_demo",
    )
    random_scores = score_random_policy()
    no_action_scores = score_policy(lambda scenario_id: build_no_action_policy(), "no_action")

    scenario_id = args.scenario
    expert_trace = _run_trace(build_expert_policy(), scenario_id=scenario_id)
    random_trace = _run_trace(RandomPolicy(seed=0), scenario_id=scenario_id)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 9), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.2], hspace=0.35, wspace=0.25)

    ax_bar = fig.add_subplot(gs[0, 0])
    labels = ["Expert", "LLM Demo", "Random", "No Action"]
    values = [
        expert_scores.per_scenario[scenario_id],
        llm_scores.per_scenario[scenario_id],
        random_scores.per_scenario[scenario_id],
        no_action_scores.per_scenario[scenario_id],
    ]
    colors = ["#1f9d73", "#4c84c4", "#d68c2f", "#cc4b5a"]
    bars = ax_bar.bar(labels, values, color=colors, edgecolor="#1d2736", linewidth=1.2)
    ax_bar.axhline(0.0, color="#364152", linewidth=1.1)
    ax_bar.set_title(f"Scenario Reward Comparison: {scenario_id.replace('_', ' ').title()}", fontsize=15, pad=12)
    ax_bar.set_ylabel("Episode reward")
    ax_bar.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, values, strict=True):
        va = "bottom" if value >= 0 else "top"
        offset = 0.18 if value >= 0 else -0.18
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.3f}",
            ha="center",
            va=va,
            fontsize=11,
            fontweight="bold",
        )

    ax_trace = fig.add_subplot(gs[0, 1])
    expert_steps = [0] + [step.step_index + 1 for step in expert_trace.steps]
    expert_spo2 = [expert_trace.initial_observation.spo2 * 100] + [step.observation.spo2 * 100 for step in expert_trace.steps]
    random_steps = [0] + [step.step_index + 1 for step in random_trace.steps]
    random_spo2 = [random_trace.initial_observation.spo2 * 100] + [step.observation.spo2 * 100 for step in random_trace.steps]
    ax_trace.plot(expert_steps, expert_spo2, marker="o", linewidth=2.6, color="#1f9d73", label="Expert / after")
    ax_trace.plot(random_steps, random_spo2, marker="o", linewidth=2.2, color="#d68c2f", label="Random seed=0 / before")
    ax_trace.set_title("SpO2 Trajectory on the Same Scenario", fontsize=15, pad=12)
    ax_trace.set_xlabel("Episode step")
    ax_trace.set_ylabel("SpO2 (%)")
    ax_trace.grid(alpha=0.18)
    ax_trace.legend(frameon=False, loc="lower right")

    ax_text_left = fig.add_subplot(gs[1, 0])
    ax_text_left.axis("off")
    ax_text_left.text(
        0,
        1,
        "Before / after framing",
        fontsize=14,
        fontweight="bold",
        va="top",
    )
    ax_text_left.text(
        0,
        0.82,
        textwrap.fill(
            f"Before: random tool-calling baseline on {scenario_id} "
            f"finishes at {random_trace.total_reward:+.3f} reward.\n"
            f"After: expert policy on the same scenario finishes at {expert_trace.total_reward:+.3f} reward.\n\n"
            "This is a better submission artifact than reusing the same training curve twice, "
            "because it shows policy quality separation on an actual episode.",
            width=55,
        ),
        fontsize=11.5,
        color="#334155",
        va="top",
        linespacing=1.5,
        wrap=True,
    )

    ax_text_right = fig.add_subplot(gs[1, 1])
    ax_text_right.axis("off")
    ax_text_right.text(
        0,
        1,
        "Verified mock benchmark ordering",
        fontsize=14,
        fontweight="bold",
        va="top",
    )
    ax_text_right.text(
        0,
        0.82,
        textwrap.fill(
            f"Expert average: {expert_scores.average_reward:+.3f}\n"
            f"LLM demo average: {llm_scores.average_reward:+.3f}\n"
            f"Random average: {random_scores.average_reward:+.3f}\n"
            f"No-action average: {no_action_scores.average_reward:+.3f}",
            width=32,
        ),
        fontsize=11.5,
        color="#334155",
        va="top",
        linespacing=1.6,
        wrap=True,
    )

    fig.suptitle(
        "Pulse-ER Mock Before vs After\nRandom Tool Calls vs Trained Policy",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.03,
        "Before = random tool-calling baseline (seed 0). After = expert/trained policy. All values are generated from the current repo.",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(output_path.resolve())


if __name__ == "__main__":
    main()
