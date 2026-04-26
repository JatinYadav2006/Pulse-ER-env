"""Submission-facing TRL GRPO training entrypoint for Pulse-ER.

This follows the official TRL OpenEnv ``environment_factory`` pattern so the
hackathon training story aligns with the organizer examples. It uses the public
Pulse-ER client and the ``PulseToolEnv`` wrapper instead of importing server
internals into the trainer.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Any

from .server.mock_scenarios import DEFAULT_MOCK_SCENARIO_ID, MOCK_SCENARIOS
from .trl_env import configure_trl_env, get_environment_factory


DEFAULT_PROMPT = (
    "You are managing a trauma patient in Pulse-ER. Use the available clinical tools to "
    "stabilize the patient, prevent deterioration, and maximize reward. Do not guess tool "
    "arguments blindly; choose interventions that match the observed physiology."
)

DEFAULT_SMOKE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_REAL_SCENARIO_ID = "polytrauma_demo"


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


def _resolve_backend_scenario(*, backend: str, scenario: str | None) -> str:
    """Resolve and validate scenario_id for the selected backend."""

    if backend == "mock":
        if not scenario:
            return DEFAULT_MOCK_SCENARIO_ID
        if scenario not in MOCK_SCENARIOS:
            valid = ", ".join(sorted(MOCK_SCENARIOS))
            raise ValueError(
                f"Unsupported mock scenario '{scenario}'. Expected one of: {valid}. "
                f"Use --backend real for Pulse scenarios such as '{DEFAULT_REAL_SCENARIO_ID}'."
            )
        return scenario

    if not scenario:
        return DEFAULT_REAL_SCENARIO_ID

    # Real scenario validation is optional because importing real scenario modules can
    # require Pulse runtime dependencies that are not always present in lightweight setups.
    try:
        from .server.scenarios import SCENARIOS
    except Exception:
        return scenario

    if scenario not in SCENARIOS:
        valid = ", ".join(sorted(SCENARIOS))
        raise ValueError(
            f"Unsupported real scenario '{scenario}'. Expected one of: {valid}. "
            f"Use --backend mock for deterministic mock scenarios such as '{DEFAULT_MOCK_SCENARIO_ID}'."
        )
    return scenario


def _resolve_generation_schedule(
    per_device_train_batch_size: int,
    num_generations: int,
) -> tuple[int, int]:
    """Make the GRPO batch/generation schedule valid and predictable.

    GRPO groups rollouts by prompt, so the generation batch size must be
    divisible by ``num_generations``. For smoke runs we prefer to preserve the
    requested number of generations and raise the batch size up to the nearest
    valid multiple instead of silently reducing exploration.
    """

    if per_device_train_batch_size <= 0:
        raise ValueError("per_device_train_batch_size must be greater than 0.")
    if num_generations < 2:
        raise ValueError("num_generations must be at least 2 for GRPO.")

    if per_device_train_batch_size % num_generations == 0:
        return per_device_train_batch_size, num_generations

    adjusted_batch_size = (
        ((per_device_train_batch_size + num_generations - 1) // num_generations)
        * num_generations
    )
    print(
        "Adjusted per_device_train_batch_size from "
        f"{per_device_train_batch_size} to {adjusted_batch_size} so it is divisible by "
        f"num_generations={num_generations}."
    )
    return adjusted_batch_size, num_generations


def _extract_metric_history(
    log_history: list[dict],
    metric_names: str | tuple[str, ...],
) -> list[dict[str, float]]:
    """Collect one scalar metric from trainer log history.

    TRL log keys vary slightly across versions. We accept a preferred metric
    name plus fallback aliases so reward artifacts keep working across local
    and Colab runs.
    """

    if isinstance(metric_names, str):
        metric_names = (metric_names,)
    points: list[dict[str, float]] = []
    for record in log_history:
        if "step" not in record:
            continue
        metric_value = None
        for metric_name in metric_names:
            if metric_name in record:
                metric_value = record[metric_name]
                break
        if metric_value is None:
            continue
        points.append({"step": float(record["step"]), "value": float(metric_value)})
    return points


def _write_metric_artifacts(output_dir: Path, log_history: list[dict]) -> None:
    """Write judge-friendly metric JSON and SVG plots from trainer logs."""

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    reward_points = _extract_metric_history(log_history, ("reward", "train/reward"))
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


def _detect_git_commit() -> str | None:
    """Return HEAD commit hash if available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def _write_run_manifest(output_dir: Path, payload: dict[str, Any]) -> None:
    """Persist run metadata so training results are reproducible."""

    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    parser.add_argument("--model", default=DEFAULT_SMOKE_MODEL)
    parser.add_argument("--env-url", default="http://127.0.0.1:8000")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--output-dir", default="outputs/pulse_er_grpo")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:  # pragma: no cover - depends on training environment
        raise RuntimeError(
            "train_grpo.py requires TRL. Install the Hugging Face TRL/OpenEnv training stack first."
        ) from exc

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on training environment
        raise RuntimeError(
            "train_grpo.py requires torch. Install the Hugging Face TRL/OpenEnv training stack first."
        ) from exc

    peft_config = None
    processing_class = None
    model = args.model

    cuda_available = bool(torch.cuda.is_available())
    use_cpu = bool(args.use_cpu or not cuda_available)
    bf16_enabled = bool(cuda_available and not use_cpu and torch.cuda.is_bf16_supported())
    fp16_enabled = bool(cuda_available and not use_cpu and not bf16_enabled)
    per_device_train_batch_size, num_generations = _resolve_generation_schedule(
        args.per_device_train_batch_size,
        args.num_generations,
    )
    scenario_id = _resolve_backend_scenario(backend=args.backend, scenario=args.scenario)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.use_qlora:
        if use_cpu:
            raise RuntimeError("QLoRA requires a CUDA GPU. Remove --use-cpu or run on a GPU-backed job.")
        try:
            from peft import LoraConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - depends on training environment
            raise RuntimeError(
                "QLoRA requires `peft`, `transformers`, and `bitsandbytes`. "
                "Install the training extras before running with --use-qlora."
            ) from exc

        compute_dtype = torch.bfloat16 if bf16_enabled else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quantization_config,
            device_map="auto",
        )
        model.config.use_cache = False
        processing_class = AutoTokenizer.from_pretrained(args.model)
        if processing_class.pad_token is None and processing_class.eos_token is not None:
            processing_class.pad_token = processing_class.eos_token

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        print(
            "Enabled QLoRA with 4-bit NF4 quantization "
            f"(r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})."
        )

    configure_trl_env(
        env_url=args.env_url,
        scenario_id=scenario_id,
        backend_kind=args.backend,
    )
    if "qwen2.5" in args.model.lower():
        print(
            "Warning: Qwen2.5 chat templates are not consistently recognized by the current "
            "TRL response-schema utility for tool-calling GRPO. Prefer a supported family such "
            "as Qwen/Qwen3-0.6B for smoke runs."
        )
    dataset = build_dataset(args.num_samples, args.prompt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    environment_factory = get_environment_factory()

    trainer = GRPOTrainer(
        model=model,
        train_dataset=dataset,
        reward_funcs=pulse_reward,
        processing_class=processing_class,
        peft_config=peft_config,
        args=GRPOConfig(
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            max_completion_length=args.max_steps,
            num_generations=num_generations,
            log_completions=True,
            use_cpu=use_cpu,
            bf16=bf16_enabled,
            fp16=fp16_enabled,
            chat_template_kwargs={"enable_thinking": False},
        ),
        environment_factory=environment_factory,
    )
    _write_run_manifest(
        output_dir,
        {
            "backend": args.backend,
            "scenario_id": scenario_id,
            "model": args.model,
            "env_url": args.env_url,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "num_generations": num_generations,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs,
            "use_qlora": args.use_qlora,
            "lora_r": args.lora_r if args.use_qlora else None,
            "lora_alpha": args.lora_alpha if args.use_qlora else None,
            "lora_dropout": args.lora_dropout if args.use_qlora else None,
            "use_cpu": use_cpu,
            "bf16": bf16_enabled,
            "fp16": fp16_enabled,
            "git_commit": _detect_git_commit(),
        },
    )
    trainer.train()
    trainer.save_state()
    _write_metric_artifacts(output_dir, trainer.state.log_history)


if __name__ == "__main__":
    main()
