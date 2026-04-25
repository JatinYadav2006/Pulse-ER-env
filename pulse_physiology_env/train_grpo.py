"""Submission-facing TRL GRPO training entrypoint for Pulse-ER.

This follows the official TRL OpenEnv ``environment_factory`` pattern so the
hackathon training story aligns with the organizer examples. It uses the public
Pulse-ER client and the ``PulseToolEnv`` wrapper instead of importing server
internals into the trainer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .trl_env import configure_trl_env, get_environment_factory


DEFAULT_PROMPT = (
    "You are managing a trauma patient in Pulse-ER. Use the available clinical tools to "
    "stabilize the patient, prevent deterioration, and maximize reward. Do not guess tool "
    "arguments blindly; choose interventions that match the observed physiology."
)


def pulse_reward(environments, **kwargs) -> list[float]:
    """Read per-episode reward from the TRL environment instances."""

    del kwargs
    return [float(env.reward) for env in environments]


def build_dataset(num_samples: int, prompt: str):
    """Create a simple prompt dataset for GRPO environment training."""

    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - depends on training environment
        raise RuntimeError(
            "train_grpo.py requires the `datasets` package. Install TRL training dependencies first."
        ) from exc

    return Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": prompt}] for _ in range(num_samples)],
            "scenario_id": [None] * num_samples,
        }
    )


def _extract_metric_history(log_history: list[dict], metric_name: str) -> list[dict[str, float]]:
    """Collect one scalar metric from trainer log history."""

    points: list[dict[str, float]] = []
    for record in log_history:
        if metric_name not in record:
            continue
        if "step" not in record:
            continue
        points.append({"step": float(record["step"]), "value": float(record[metric_name])})
    return points


def _write_metric_artifacts(output_dir: Path, log_history: list[dict]) -> None:
    """Write judge-friendly metric JSON and SVG plots from trainer logs."""

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    reward_points = _extract_metric_history(log_history, "train/reward")
    loss_points = _extract_metric_history(log_history, "loss")

    (metrics_dir / "reward_history.json").write_text(json.dumps(reward_points, indent=2), encoding="utf-8")
    (metrics_dir / "loss_history.json").write_text(json.dumps(loss_points, indent=2), encoding="utf-8")

    _write_svg_plot(
        metrics_dir / "reward_curve.svg",
        title="Pulse-ER GRPO Reward Curve",
        x_label="Training Step",
        y_label="Average Reward",
        points=reward_points,
        line_color="#198754",
    )
    _write_svg_plot(
        metrics_dir / "loss_curve.svg",
        title="Pulse-ER GRPO Loss Curve",
        x_label="Training Step",
        y_label="Loss",
        points=loss_points,
        line_color="#0d6efd",
    )


def _write_svg_plot(
    output_path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    points: list[dict[str, float]],
    line_color: str,
) -> None:
    """Render a small standalone SVG line chart without extra dependencies."""

    width = 760
    height = 420
    margin_left = 80
    margin_right = 24
    margin_top = 52
    margin_bottom = 64
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    if not points:
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width/2}" y="{height/2}" text-anchor="middle" font-size="18" font-family="Arial">No data for {title}</text>
</svg>"""
        output_path.write_text(svg, encoding="utf-8")
        return

    x_values = [point["step"] for point in points]
    y_values = [point["value"] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_max == x_min:
        x_max += 1.0
    if y_max == y_min:
        y_max += 1.0

    def map_x(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * plot_width

    def map_y(value: float) -> float:
        return margin_top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    polyline = " ".join(f"{map_x(point['step']):.1f},{map_y(point['value']):.1f}" for point in points)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-family="Arial" font-weight="bold">{title}</text>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="2" />
<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="2" />
<polyline fill="none" stroke="{line_color}" stroke-width="3" points="{polyline}" />
<text x="{width/2}" y="{height - 16}" text-anchor="middle" font-size="14" font-family="Arial">{x_label}</text>
<text x="24" y="{height/2}" text-anchor="middle" font-size="14" font-family="Arial" transform="rotate(-90 24 {height/2})">{y_label}</text>
<text x="{margin_left}" y="{margin_top + plot_height + 24}" text-anchor="start" font-size="12" font-family="Arial">{x_min:.0f}</text>
<text x="{margin_left + plot_width}" y="{margin_top + plot_height + 24}" text-anchor="end" font-size="12" font-family="Arial">{x_max:.0f}</text>
<text x="{margin_left - 12}" y="{margin_top + plot_height}" text-anchor="end" font-size="12" font-family="Arial">{y_min:.2f}</text>
<text x="{margin_left - 12}" y="{margin_top}" text-anchor="end" font-size="12" font-family="Arial">{y_max:.2f}</text>
</svg>"""
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="mock", choices=("mock", "real"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--env-url", default="http://127.0.0.1:8000")
    parser.add_argument("--scenario", default="polytrauma_demo")
    parser.add_argument("--output-dir", default="outputs/pulse_er_grpo")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:  # pragma: no cover - depends on training environment
        raise RuntimeError(
            "train_grpo.py requires TRL. Install the Hugging Face TRL/OpenEnv training stack first."
        ) from exc

    configure_trl_env(
        env_url=args.env_url,
        scenario_id=args.scenario,
        backend_kind=args.backend,
    )
    dataset = build_dataset(args.num_samples, args.prompt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    environment_factory = get_environment_factory()

    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=dataset,
        reward_funcs=pulse_reward,
        args=GRPOConfig(
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            max_completion_length=args.max_steps,
            num_generations=args.num_generations,
            log_completions=True,
            chat_template_kwargs={"enable_thinking": False},
        ),
        environment_factory=environment_factory,
    )
    trainer.train()
    trainer.save_state()
    _write_metric_artifacts(output_dir, trainer.state.log_history)


if __name__ == "__main__":
    main()
