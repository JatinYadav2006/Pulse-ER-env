"""HF Space dashboard data and HTML for Pulse-ER."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from pulse_physiology_env.demo_llm_policy import heuristic_infer_fn
from pulse_physiology_env.episode_runner import EpisodeRunner, EpisodeTrace
from pulse_physiology_env.eval_mock import score_policy, score_random_policy
from pulse_physiology_env.policies import LLMPolicy, RandomPolicy, build_expert_policy, build_no_action_policy
from pulse_physiology_env.server.adapters import KNOWN_TOOL_NAMES, MockPulseAdapter
from pulse_physiology_env.server.pathology_architect import PathologyArchitect
from pulse_physiology_env.server.tools import AVAILABLE_TOOL_NAMES, CLINICAL_45_TOOL_NAMES

DEFAULT_SPACE_SCENARIO = "respiratory_distress"
DEFAULT_SPACE_POLICY = "expert"
RUNTIME_SETTINGS = {
    "observation_noise_level": 0.3,
    "time_pressure_enabled": True,
    "seed": 0,
    "max_steps": 8,
}

POLICY_META: dict[str, dict[str, str]] = {
    "expert": {
        "label": "Expert",
        "summary": "Deterministic-plus-adaptive baseline that follows authored playbooks and trauma heuristics.",
    },
    "llm_demo": {
        "label": "LLM Demo",
        "summary": "Prompt-driven heuristic baseline that is stronger than random but still inconsistent on harder scenarios.",
    },
    "random": {
        "label": "Random",
        "summary": "Uninformed tool caller that samples valid actions without clinical sequencing.",
    },
    "no_action": {
        "label": "No Action",
        "summary": "Passive baseline that mostly advances time and lets the patient deteriorate untreated.",
    },
}

SPACE_SCENARIO_META: dict[str, dict[str, str]] = {
    "baseline_stable": {
        "label": "Baseline Stable",
        "title": "Baseline Stability Console",
        "summary": (
            "A low-acuity case that exposes policy restraint. The useful signal here is not dramatic rescue, "
            "but avoiding needless interventions while keeping the patient stable under noisy observations."
        ),
        "teaching_point": "assess first and avoid over-treatment in stable physiology",
    },
    "respiratory_distress": {
        "label": "Respiratory Distress",
        "title": "Respiratory Rescue Console",
        "summary": (
            "A reproducible respiratory deterioration case under observation noise and time pressure. "
            "The replay shows how the trained policy restores oxygenation and lowers work of breathing through ordered support."
        ),
        "teaching_point": "restore oxygenation early and do not wait through worsening hypoxemia",
    },
    "hemorrhagic_shock": {
        "label": "Hemorrhagic Shock",
        "title": "Hemorrhagic Shock Console",
        "summary": (
            "A circulation-first case where delayed action is punished quickly. "
            "This view is useful because it shows how strongly the environment separates hemorrhage control from passive or noisy policies."
        ),
        "teaching_point": "treat shock early, because delay compounds deterioration",
    },
}

TOOL_GROUPS: list[dict[str, Any]] = [
    {
        "name": "Assessment",
        "summary": "Initial reads, triage summaries, and deterioration checks.",
        "tools": [
            "get_vitals",
            "get_respiratory_status",
            "check_deterioration",
            "summarize_state",
            "recommend_next_step",
        ],
    },
    {
        "name": "Airway / Breathing",
        "summary": "Respiratory support, decompression, and patient positioning.",
        "tools": [
            "give_oxygen",
            "airway_support",
            "needle_decompression",
            "position_patient",
        ],
    },
    {
        "name": "Circulation",
        "summary": "Hemorrhage control, fluids, and pressor escalation.",
        "tools": [
            "control_bleeding",
            "give_fluids",
            "give_pressor",
        ],
    },
    {
        "name": "Diagnostics",
        "summary": "Delayed labs that become available after simulated time passes.",
        "tools": [
            "get_blood_gas",
            "get_cbc",
            "get_bmp",
        ],
    },
    {
        "name": "Procedures / Flow",
        "summary": "Procedural intervention and time control inside the episode loop.",
        "tools": [
            "pericardiocentesis",
            "advance_time",
        ],
    },
]


def _patient_count() -> int:
    return len(PathologyArchitect().supported_patients())


def _injury_count() -> int:
    return len(PathologyArchitect().supported_injury_types())


def _combo_count() -> int:
    return len(PathologyArchitect().default_injury_combos())


def _repo_url() -> str:
    return os.environ.get("PULSE_ER_REPO_URL", "https://github.com/JatinYadav2006/Pulse-ER-env.git")


def _training_link() -> str:
    return os.environ.get(
        "PULSE_ER_TRAINING_LINK",
        "https://github.com/JatinYadav2006/Pulse-ER-env/blob/kumarthegoat/train_grpo.py",
    )


def _trace_to_payload(trace: EpisodeTrace) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    initial = trace.initial_observation
    frames.append(
        {
            "step_index": -1,
            "tool_name": "reset",
            "message": f"Scenario loaded: {trace.scenario_id}",
            "reward": 0.0,
            "sim_time_s": initial.sim_time_s,
            "heart_rate_bpm": initial.heart_rate_bpm,
            "systolic_bp_mmhg": initial.systolic_bp_mmhg,
            "diastolic_bp_mmhg": initial.diastolic_bp_mmhg,
            "spo2": initial.spo2,
            "respiration_rate_bpm": initial.respiration_rate_bpm,
            "blood_volume_ml": initial.blood_volume_ml,
            "mental_status": getattr(initial.mental_status, "value", initial.mental_status),
            "active_alerts": list(initial.active_alerts),
        }
    )
    for step in trace.steps:
        obs = step.observation
        frames.append(
            {
                "step_index": step.step_index,
                "tool_name": step.action.tool_name,
                "message": (
                    step.tool_result["message"]
                    if step.tool_result is not None and "message" in step.tool_result
                    else step.error["message"]
                    if step.error is not None and "message" in step.error
                    else step.action.tool_name
                ),
                "reward": step.reward,
                "sim_time_s": obs.sim_time_s,
                "heart_rate_bpm": obs.heart_rate_bpm,
                "systolic_bp_mmhg": obs.systolic_bp_mmhg,
                "diastolic_bp_mmhg": obs.diastolic_bp_mmhg,
                "spo2": obs.spo2,
                "respiration_rate_bpm": obs.respiration_rate_bpm,
                "blood_volume_ml": obs.blood_volume_ml,
                "mental_status": getattr(obs.mental_status, "value", obs.mental_status),
                "active_alerts": list(obs.active_alerts),
            }
        )

    final_summary = trace.summary()
    return {
        "scenario_id": trace.scenario_id,
        "policy_name": trace.policy_name,
        "summary": final_summary,
        "events": list(trace.events),
        "frames": frames,
        "action_log": [
            {
                "step_index": step.step_index,
                "tool_name": step.action.tool_name,
                "message": step.tool_result["message"] if step.tool_result else step.action.tool_name,
                "reward": step.reward,
            }
            for step in trace.steps
        ],
        "config": dict(RUNTIME_SETTINGS),
    }


@lru_cache(maxsize=1)
def get_policy_benchmark_payload() -> dict[str, Any]:
    expert = score_policy(lambda scenario_id: build_expert_policy(), "expert")
    llm_demo = score_policy(
        lambda scenario_id: LLMPolicy(infer_fn=heuristic_infer_fn, name="llm_demo"),
        "llm_demo",
    )
    random_policy = score_random_policy()
    no_action = score_policy(lambda scenario_id: build_no_action_policy(), "no_action")

    comparison = [
        {"label": "Expert", "value": expert.average_reward, "status": "good"},
        {
            "label": "LLM Demo",
            "value": llm_demo.average_reward,
            "status": "good" if llm_demo.average_reward > 0 else "warn",
        },
        {"label": "Random", "value": random_policy.average_reward, "status": "bad"},
        {"label": "No Action", "value": no_action.average_reward, "status": "bad"},
    ]
    per_scenario = {
        "expert": expert.per_scenario,
        "llm_demo": llm_demo.per_scenario,
        "random": random_policy.per_scenario,
        "no_action": no_action.per_scenario,
    }
    return {"comparison": comparison, "per_scenario": per_scenario}


def _available_space_scenarios(benchmarks: dict[str, Any]) -> list[str]:
    available = list(benchmarks["per_scenario"]["expert"].keys())
    ordered = [scenario_id for scenario_id in SPACE_SCENARIO_META if scenario_id in available]
    return ordered + [scenario_id for scenario_id in available if scenario_id not in ordered]


def _normalize_space_scenario(scenario_id: str | None, benchmarks: dict[str, Any]) -> str:
    available = _available_space_scenarios(benchmarks)
    if scenario_id in available:
        return scenario_id
    return DEFAULT_SPACE_SCENARIO if DEFAULT_SPACE_SCENARIO in available else available[0]


def _available_policy_names() -> list[str]:
    return list(POLICY_META.keys())


def _normalize_policy_name(policy_name: str | None) -> str:
    if policy_name in POLICY_META:
        return policy_name
    return DEFAULT_SPACE_POLICY


def _build_policy(policy_name: str):
    if policy_name == "expert":
        return build_expert_policy()
    if policy_name == "llm_demo":
        return LLMPolicy(infer_fn=heuristic_infer_fn, name="llm_demo")
    if policy_name == "random":
        return RandomPolicy(seed=RUNTIME_SETTINGS["seed"])
    if policy_name == "no_action":
        return build_no_action_policy()
    raise ValueError(f"Unsupported policy: {policy_name}")


@lru_cache(maxsize=32)
def get_demo_episode_payload(
    scenario_id: str = DEFAULT_SPACE_SCENARIO,
    policy_name: str = DEFAULT_SPACE_POLICY,
) -> dict[str, Any]:
    backend = MockPulseAdapter(
        default_scenario_id=scenario_id,
        observation_noise_level=RUNTIME_SETTINGS["observation_noise_level"],
        time_pressure_enabled=RUNTIME_SETTINGS["time_pressure_enabled"],
        seed=RUNTIME_SETTINGS["seed"],
    )
    runner = EpisodeRunner(backend=backend, max_steps=RUNTIME_SETTINGS["max_steps"])
    policy = _build_policy(_normalize_policy_name(policy_name))
    try:
        trace = runner.run(policy=policy, scenario_id=scenario_id)
    finally:
        close_method = getattr(backend, "close", None)
        if callable(close_method):
            close_method()
    return _trace_to_payload(trace)


def _build_primary_scenario(scenario_id: str, demo: dict[str, Any], benchmarks: dict[str, Any]) -> dict[str, str]:
    meta = SPACE_SCENARIO_META.get(
        scenario_id,
        {
            "label": scenario_id.replace("_", " ").title(),
            "title": f"{scenario_id.replace('_', ' ').title()} Console",
            "summary": "Deterministic mock replay for the selected scenario.",
            "teaching_point": "follow the measured physiology signal",
        },
    )
    frames = demo["frames"]
    first_observed = next((frame for frame in frames if frame["spo2"] is not None), frames[0])
    final = demo["summary"]
    expert_rr = benchmarks["per_scenario"]["expert"][scenario_id]
    no_action_rr = benchmarks["per_scenario"]["no_action"][scenario_id]
    selected_policy = demo["policy_name"]
    selected_label = POLICY_META.get(selected_policy, {}).get("label", selected_policy.replace("_", " ").title())
    return {
        "title": meta["title"],
        "tag": "EVALUATION CONSOLE",
        "summary": meta["summary"],
        "teaching_point": meta["teaching_point"],
        "naive_outcome": (
            f"No-action baseline on {scenario_id}: {no_action_rr:+.3f} reward with progressive deterioration."
        ),
        "trained_outcome": (
            f"Expert replay improves SpO2 from {first_observed['spo2'] * 100:.1f}% "
            f"to {final['spo2_percent']:.1f}% and finishes at {expert_rr:+.3f} reward."
        ),
        "selected_outcome": (
            f"{selected_label} on {scenario_id} ended with {final['termination_reason']} at "
            f"{final['total_reward']:+.3f} reward after {final['num_steps']} steps."
        ),
    }


def _build_research_highlights(benchmarks: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"label": f"{len(KNOWN_TOOL_NAMES)} public tools", "value": "consumer contract exposed in the Space"},
        {"label": f"{len(CLINICAL_45_TOOL_NAMES)} clinical tools", "value": "full clinical intervention surface in the real runtime"},
        {"label": f"{len(AVAILABLE_TOOL_NAMES)} runtime names", "value": "combined real backend execution/alias surface"},
        {"label": f"{_patient_count()} patient profiles", "value": "supported by PathologyArchitect"},
    ]


def _build_tool_surface() -> dict[str, Any]:
    return {
        "public_contract_count": len(KNOWN_TOOL_NAMES),
        "clinical_surface_count": len(CLINICAL_45_TOOL_NAMES),
        "runtime_name_count": len(AVAILABLE_TOOL_NAMES),
        "groups": TOOL_GROUPS,
    }


def _build_engine_layers() -> list[dict[str, str]]:
    return [
        {
            "title": "Physiology Core",
            "value": "Pulse 4.3.2",
            "detail": "Real backend runs on the Pulse physiology engine rather than a hand-scripted simulator.",
        },
        {
            "title": "Reward Engine",
            "value": "Safety-shaped",
            "detail": "Dense reward, sequencing penalties, and terminal scoring drive clinically ordered behavior.",
        },
        {
            "title": "ATLS Judge",
            "value": "Protocol scoring",
            "detail": "Action history can be graded into human-readable pass/fail protocol checks.",
        },
        {
            "title": "Runtime Effects",
            "value": "Noise + time pressure",
            "detail": "Observation perturbations and deterioration pressure make the policy work under uncertainty.",
        },
        {
            "title": "PathologyArchitect",
            "value": f"{_injury_count()} injury families",
            "detail": f"{_patient_count()} baseline patients and {_combo_count()} default combo ladders support generated trauma cases.",
        },
        {
            "title": "Adversarial Evaluation",
            "value": "Breaking points",
            "detail": "Policies can be stress-tested with stacked injuries rather than judged on single static prompts.",
        },
    ]


def _build_benchmark_matrix(benchmarks: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_id in _available_space_scenarios(benchmarks):
        rows.append(
            {
                "scenario_id": scenario_id,
                "label": SPACE_SCENARIO_META.get(scenario_id, {}).get("label", scenario_id.replace("_", " ").title()),
                "expert": benchmarks["per_scenario"]["expert"][scenario_id],
                "llm_demo": benchmarks["per_scenario"]["llm_demo"][scenario_id],
                "random": benchmarks["per_scenario"]["random"][scenario_id],
                "no_action": benchmarks["per_scenario"]["no_action"][scenario_id],
            }
        )
    return rows


def _build_runtime_profile(demo: dict[str, Any]) -> list[dict[str, str]]:
    summary = demo["summary"]
    config = demo["config"]
    return [
        {"label": "Seed", "value": str(config["seed"])},
        {"label": "Noise", "value": f"{config['observation_noise_level']:.1f}"},
        {"label": "Time pressure", "value": "enabled" if config["time_pressure_enabled"] else "disabled"},
        {"label": "Step budget", "value": str(config["max_steps"])},
        {"label": "Final mental status", "value": str(summary["mental_status"])},
        {"label": "Alert count", "value": str(len(summary["active_alerts"]))},
    ]


def _build_policy_outcome(demo: dict[str, Any]) -> list[dict[str, str]]:
    summary = demo["summary"]
    systolic = summary["systolic_bp_mmhg"]
    diastolic = summary["diastolic_bp_mmhg"]
    map_proxy = (
        f"{round((systolic + 2 * diastolic) / 3)} mmHg"
        if systolic is not None and diastolic is not None
        else "n/a"
    )
    return [
        {"label": "Termination", "value": summary["termination_reason"].replace("_", " ")},
        {"label": "Total reward", "value": f"{summary['total_reward']:+.3f}"},
        {"label": "Steps executed", "value": str(summary["num_steps"])},
        {"label": "Final SpO2", "value": f"{summary['spo2_percent']:.1f}%" if summary["spo2_percent"] is not None else "n/a"},
        {"label": "Final MAP proxy", "value": map_proxy},
        {"label": "Final alerts", "value": str(len(summary["active_alerts"]))},
    ]


def get_dashboard_payload(scenario_id: str | None = None, policy_name: str | None = None) -> dict[str, Any]:
    benchmarks = get_policy_benchmark_payload()
    selected_scenario = _normalize_space_scenario(scenario_id, benchmarks)
    selected_policy = _normalize_policy_name(policy_name)
    demo = get_demo_episode_payload(selected_scenario, selected_policy)
    scenario = _build_primary_scenario(selected_scenario, demo, benchmarks)
    return {
        "hero": {
            "title": "Pulse-ER",
            "subtitle": "Physiology Evaluation Console",
            "description": (
                "A trauma reinforcement-learning environment backed by Pulse physiology. "
                "This Space exposes more than a single replay: scenario controls, policy execution, "
                "tool surface, runtime profile, and evaluation stack are all surfaced directly so the environment reads like a real benchmark console."
            ),
            "badges": [
                f"{len(KNOWN_TOOL_NAMES)}-tool public contract",
                f"{len(CLINICAL_45_TOOL_NAMES)}-tool clinical surface",
                "Pulse 4.3.2 runtime",
            ],
        },
        "selected_scenario": selected_scenario,
        "selected_policy": selected_policy,
        "available_scenarios": [
            {
                "id": scenario_key,
                "label": SPACE_SCENARIO_META.get(scenario_key, {}).get("label", scenario_key.replace("_", " ").title()),
            }
            for scenario_key in _available_space_scenarios(benchmarks)
        ],
        "available_policies": [
            {"id": policy_key, "label": POLICY_META[policy_key]["label"], "summary": POLICY_META[policy_key]["summary"]}
            for policy_key in _available_policy_names()
        ],
        "scenario": scenario,
        "policy_comparison": benchmarks["comparison"],
        "benchmark_matrix": _build_benchmark_matrix(benchmarks),
        "research_highlights": _build_research_highlights(benchmarks),
        "tool_surface": _build_tool_surface(),
        "engine_layers": _build_engine_layers(),
        "runtime_profile": _build_runtime_profile(demo),
        "policy_outcome": _build_policy_outcome(demo),
        "demo_episode": demo,
        "links": {
            "repo_url": _repo_url(),
            "training_url": _training_link(),
        },
    }


def build_dashboard_html() -> str:
    payload = json.dumps(get_dashboard_payload())
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Pulse-ER Physiology Evaluation Console</title>
  <style>
    :root {{
      --bg: #08111f;
      --panel: rgba(15, 24, 41, 0.92);
      --panel-strong: rgba(10, 17, 30, 0.98);
      --panel-soft: rgba(28, 41, 66, 0.35);
      --border: rgba(118, 154, 219, 0.2);
      --text: #eef4ff;
      --muted: #93a7c8;
      --cyan: #5fd8ff;
      --teal: #47e5bb;
      --amber: #ffc35a;
      --red: #ff7388;
      --green: #5af0a5;
      --violet: #9b8cff;
      --shadow: 0 24px 60px rgba(0, 0, 0, 0.34);
      --font: "Segoe UI", "IBM Plex Sans", "Helvetica Neue", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: var(--font);
      color: var(--text);
      background:
        radial-gradient(circle at 12% 15%, rgba(95, 216, 255, 0.16), transparent 22%),
        radial-gradient(circle at 86% 0%, rgba(71, 229, 187, 0.12), transparent 22%),
        linear-gradient(180deg, #07101d 0%, #08111f 100%);
    }}
    .shell {{
      width: min(1540px, calc(100vw - 28px));
      margin: 20px auto 36px;
      display: grid;
      gap: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 26px;
      box-shadow: var(--shadow);
      overflow: hidden;
      backdrop-filter: blur(18px);
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
    }}
    .hero-main {{
      padding: 30px 32px 28px;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      color: var(--cyan);
      font-weight: 800;
      font-size: 12px;
    }}
    .hero-title {{
      margin: 12px 0 8px;
      font-size: clamp(38px, 5vw, 60px);
      line-height: 0.94;
      letter-spacing: -0.05em;
    }}
    .hero-subtitle {{
      margin: 0 0 14px;
      font-size: clamp(20px, 2vw, 28px);
      color: #d9e7ff;
      font-weight: 650;
    }}
    .hero-copy {{
      margin: 0;
      max-width: 68ch;
      color: var(--muted);
      line-height: 1.68;
      font-size: 15px;
    }}
    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 22px;
    }}
    .badge {{
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(76, 106, 163, 0.14);
      border: 1px solid rgba(124, 160, 224, 0.22);
      font-size: 13px;
      font-weight: 650;
    }}
    .hero-side {{
      padding: 24px;
      display: grid;
      gap: 12px;
      align-content: start;
    }}
    .section-label {{
      display: inline-flex;
      align-items: center;
      width: fit-content;
      padding: 7px 12px;
      border-radius: 999px;
      background: rgba(255, 195, 90, 0.12);
      color: var(--amber);
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-size: 12px;
      font-weight: 800;
    }}
    .layer-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .layer-card {{
      padding: 15px 16px;
      border-radius: 18px;
      background: var(--panel-strong);
      border: 1px solid rgba(255,255,255,0.06);
    }}
    .layer-title {{
      color: #dfeaff;
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 4px;
    }}
    .layer-value {{
      font-size: 22px;
      font-weight: 800;
      letter-spacing: -0.03em;
      margin-bottom: 6px;
    }}
    .layer-detail {{
      color: var(--muted);
      font-size: 12.5px;
      line-height: 1.5;
    }}
    .main-grid {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
    }}
    .stack {{
      display: grid;
      gap: 18px;
    }}
    .section {{
      padding: 22px 24px 24px;
    }}
    .section-header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 18px;
    }}
    .section-title {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0.02em;
    }}
    .section-meta {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}
    .scenario-head {{
      display: grid;
      gap: 10px;
      margin-bottom: 18px;
    }}
    .scenario-title {{
      margin: 0;
      font-size: 30px;
      line-height: 1.06;
      letter-spacing: -0.04em;
    }}
    .scenario-copy {{
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
      font-size: 14px;
    }}
    .scenario-selector {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .scenario-chip {{
      appearance: none;
      border: 1px solid rgba(109, 149, 220, 0.22);
      background: rgba(90, 115, 169, 0.12);
      color: #dcecff;
      padding: 10px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      cursor: pointer;
      transition: transform 140ms ease, border-color 140ms ease, background 140ms ease;
    }}
    .scenario-chip:hover {{
      transform: translateY(-1px);
      border-color: rgba(87, 214, 255, 0.45);
    }}
    .scenario-chip.active {{
      background: linear-gradient(135deg, rgba(87, 214, 255, 0.26), rgba(71, 229, 187, 0.16));
      border-color: rgba(87, 214, 255, 0.5);
      color: #f3fbff;
    }}
    .scenario-contrast {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .contrast-card {{
      padding: 15px 16px;
      border-radius: 18px;
      background: var(--panel-strong);
      border: 1px solid rgba(255,255,255,0.06);
    }}
    .contrast-card strong {{
      display: block;
      margin-bottom: 6px;
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }}
    .contrast-card.bad strong {{ color: var(--red); }}
    .contrast-card.good strong {{ color: var(--green); }}
    .teaching {{
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.05);
      background: rgba(255, 195, 90, 0.08);
      color: #f4ddab;
      line-height: 1.55;
      font-size: 13px;
    }}
    .score-strip {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }}
    .score-card {{
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
    }}
    .score-card.active {{
      background: linear-gradient(135deg, rgba(95, 216, 255, 0.12), rgba(71, 229, 187, 0.08));
      border-color: rgba(95, 216, 255, 0.3);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
    }}
    .score-card .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      margin-bottom: 8px;
    }}
    .score-card .value {{
      font-size: 24px;
      font-weight: 800;
      letter-spacing: -0.03em;
    }}
    .good {{ color: var(--green); }}
    .warn {{ color: var(--amber); }}
    .bad {{ color: var(--red); }}
    .monitor-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .tile {{
      padding: 16px 18px;
      border-radius: 18px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
      min-height: 98px;
    }}
    .tile-label {{
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
    }}
    .tile-value {{
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .tile-trend {{
      margin-top: 10px;
      font-size: 13px;
      color: #bed4f5;
    }}
    .trend-chart {{
      margin-top: 18px;
      padding: 16px;
      border-radius: 20px;
      background: rgba(8, 14, 27, 0.76);
      border: 1px solid rgba(255,255,255,0.05);
    }}
    .chart-svg {{
      width: 100%;
      height: 220px;
      display: block;
    }}
    .legend {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      font-size: 12px;
      color: var(--muted);
      margin-top: 10px;
    }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 8px;
      vertical-align: middle;
    }}
    .legend .spo2::before {{ background: var(--teal); }}
    .legend .hr::before {{ background: var(--cyan); }}
    .legend .reward::before {{ background: var(--amber); }}
    .scrubber {{
      display: grid;
      gap: 10px;
      margin-top: 16px;
    }}
    .scrubber-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: #4fd9ff;
    }}
    .frame-card {{
      padding: 16px;
      border-radius: 18px;
      background: var(--panel-strong);
      border: 1px solid rgba(255,255,255,0.06);
    }}
    .frame-tool {{
      font-size: 14px;
      font-weight: 800;
      margin-bottom: 8px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .frame-message {{
      color: var(--muted);
      line-height: 1.6;
      font-size: 13px;
      margin-bottom: 10px;
    }}
    .frame-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .pill {{
      padding: 7px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 650;
      background: rgba(255,255,255,0.05);
      color: #dce9ff;
    }}
    .log-list {{
      display: grid;
      gap: 10px;
      max-height: 410px;
      overflow: auto;
      padding-right: 6px;
    }}
    .log-item {{
      padding: 14px;
      border-radius: 16px;
      background: rgba(255,255,255,0.03);
      border: 1px solid transparent;
      transition: border-color 160ms ease, transform 160ms ease;
    }}
    .log-item.active {{
      border-color: rgba(87, 214, 255, 0.4);
      transform: translateX(2px);
    }}
    .log-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 8px;
    }}
    .log-tool {{
      font-size: 14px;
      font-weight: 700;
    }}
    .log-reward {{
      font-size: 12px;
      padding: 5px 8px;
      border-radius: 999px;
      background: rgba(255,255,255,0.06);
      color: #dcecff;
    }}
    .log-message {{
      font-size: 13px;
      line-height: 1.55;
      color: var(--muted);
    }}
    .runtime-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .runtime-card {{
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
    }}
    .runtime-card .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      margin-bottom: 8px;
    }}
    .runtime-card .value {{
      font-size: 20px;
      font-weight: 750;
      letter-spacing: -0.02em;
    }}
    .tool-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .tool-card {{
      padding: 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
    }}
    .tool-card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 8px;
    }}
    .tool-card-title {{
      font-size: 15px;
      font-weight: 800;
    }}
    .tool-card-count {{
      color: var(--cyan);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-weight: 800;
    }}
    .tool-card-copy {{
      color: var(--muted);
      line-height: 1.55;
      font-size: 13px;
      margin-bottom: 12px;
    }}
    .tool-chip-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .tool-chip {{
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(95, 216, 255, 0.08);
      border: 1px solid rgba(95, 216, 255, 0.14);
      color: #dcecff;
      font-size: 12px;
      font-family: "IBM Plex Mono", "Consolas", monospace;
    }}
    .matrix-wrapper {{
      overflow: auto;
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(8, 14, 27, 0.76);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 640px;
    }}
    th, td {{
      padding: 14px 16px;
      text-align: left;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      font-size: 13px;
    }}
    th {{
      color: #dce7fb;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 12px;
      background: rgba(255,255,255,0.03);
    }}
    td {{
      color: var(--muted);
    }}
    td strong {{
      color: var(--text);
    }}
    .footer-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 18px;
    }}
    .btn {{
      appearance: none;
      border: 0;
      cursor: pointer;
      padding: 13px 16px;
      border-radius: 14px;
      font-weight: 700;
      font-size: 14px;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      transition: transform 160ms ease;
    }}
    .btn:hover {{ transform: translateY(-1px); }}
    .btn.primary {{
      background: linear-gradient(135deg, var(--cyan), #22b9ff);
      color: #041221;
    }}
    .btn.secondary {{
      background: rgba(255,255,255,0.05);
      color: var(--text);
      border: 1px solid rgba(255,255,255,0.08);
    }}
    .microcopy {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }}
    @media (max-width: 1280px) {{
      .hero, .main-grid {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 880px) {{
      .monitor-grid, .score-strip, .tool-grid, .layer-grid, .runtime-grid, .scenario-contrast {{
        grid-template-columns: 1fr;
      }}
      .shell {{
        width: min(100vw - 16px, 100%);
        margin: 10px auto 24px;
      }}
      .hero-main, .hero-side, .section {{
        padding: 18px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell" id="app"></div>
  <script>
    const initialPayload = {payload};
    let state = initialPayload;
    let currentFrameIndex = 0;

    function fmtSigned(value, digits = 2) {{
      const prefix = value > 0 ? "+" : "";
      return `${{prefix}}${{Number(value).toFixed(digits)}}`;
    }}

    function formatSpo2(value) {{
      if (value === null || value === undefined) return "n/a";
      return `${{Math.round(value * 100)}}%`;
    }}

    function formatBpm(value) {{
      if (value === null || value === undefined) return "n/a";
      return `${{Math.round(value)}} bpm`;
    }}

    function formatBp(sys, dia) {{
      if (sys === null || sys === undefined || dia === null || dia === undefined) return "n/a";
      return `${{Math.round(sys)}}/${{Math.round(dia)}}`;
    }}

    function metricStatus(kind, value) {{
      if (value === null || value === undefined) return "warn";
      if (kind === "spo2") return value < 0.9 ? "bad" : value < 0.95 ? "warn" : "good";
      if (kind === "hr") return value > 120 ? "bad" : value > 100 ? "warn" : "good";
      if (kind === "rr") return value > 28 ? "bad" : value > 20 ? "warn" : "good";
      return "good";
    }}

    function buildChart(frames) {{
      const width = 680;
      const height = 240;
      const pad = 24;
      const innerWidth = width - pad * 2;
      const innerHeight = height - pad * 2;
      const domainX = frames.map((f, idx) => idx);
      const hrValues = frames.map(f => f.heart_rate_bpm ?? 0);
      const spo2Values = frames.map(f => (f.spo2 ?? 0) * 100);
      const rewardValues = frames.map(f => f.reward ?? 0);
      const xMax = Math.max(1, domainX[domainX.length - 1] || 1);
      const yMin = Math.min(...hrValues, ...spo2Values, ...rewardValues, 0);
      const yMax = Math.max(...hrValues, ...spo2Values, ...rewardValues, 120);
      const toX = (index) => pad + (index / xMax) * innerWidth;
      const toY = (value) => pad + innerHeight - ((value - yMin) / (yMax - yMin || 1)) * innerHeight;
      const poly = (values) => values.map((value, idx) => `${{toX(idx)}},${{toY(value)}}`).join(" ");
      const focusX = toX(currentFrameIndex);
      return `
        <svg class="chart-svg" viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Episode telemetry chart">
          <rect x="0" y="0" width="${{width}}" height="${{height}}" rx="18" fill="transparent"></rect>
          <line x1="${{pad}}" y1="${{height - pad}}" x2="${{width - pad}}" y2="${{height - pad}}" stroke="rgba(255,255,255,0.12)" />
          <line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height - pad}}" stroke="rgba(255,255,255,0.12)" />
          <polyline fill="none" stroke="#57d6ff" stroke-width="3" points="${{poly(hrValues)}}" />
          <polyline fill="none" stroke="#47e5bb" stroke-width="3" points="${{poly(spo2Values)}}" />
          <polyline fill="none" stroke="#ffc35a" stroke-width="2.5" stroke-dasharray="8 7" points="${{poly(rewardValues)}}" />
          <line x1="${{focusX}}" y1="${{pad}}" x2="${{focusX}}" y2="${{height - pad}}" stroke="rgba(255,255,255,0.22)" stroke-dasharray="5 5" />
        </svg>
      `;
    }}

    async function loadDashboard(scenarioId = state.selected_scenario, policyName = state.selected_policy) {{
      const query = new URLSearchParams({{
        scenario_id: scenarioId,
        policy_name: policyName,
      }});
      const response = await fetch(`/space/api/dashboard?${{query.toString()}}`);
      if (!response.ok) {{
        throw new Error(`Failed to load dashboard state for ${{scenarioId}} / ${{policyName}}`);
      }}
      state = await response.json();
      currentFrameIndex = 0;
      render();
    }}

    function render(frameIndex = currentFrameIndex) {{
      currentFrameIndex = frameIndex;
      const root = document.getElementById("app");
      const payload = state;
      const demo = payload.demo_episode;
      const frames = demo.frames;
      const currentFrame = frames[Math.min(frameIndex, frames.length - 1)];
      const previousFrame = frames[Math.max(0, Math.min(frameIndex - 1, frames.length - 1))];
      const selectedRow = payload.benchmark_matrix.find(row => row.scenario_id === payload.selected_scenario);
      const scenarioButtons = payload.available_scenarios.map((item) => `
        <button class="scenario-chip ${{item.id === payload.selected_scenario ? "active" : ""}}" data-scenario-id="${{item.id}}">
          ${{item.label}}
        </button>
      `).join("");
      const policyButtons = payload.available_policies.map((item) => `
        <button class="scenario-chip ${{item.id === payload.selected_policy ? "active" : ""}}" data-policy-id="${{item.id}}" title="${{item.summary}}">
          ${{item.label}}
        </button>
      `).join("");
      const selectedPolicyMeta = payload.available_policies.find(item => item.id === payload.selected_policy);
      const scoreStrip = [
        {{ label: "Expert", value: selectedRow.expert, selected: payload.selected_policy === "expert" }},
        {{ label: "LLM Demo", value: selectedRow.llm_demo, selected: payload.selected_policy === "llm_demo" }},
        {{ label: "Random", value: selectedRow.random, selected: payload.selected_policy === "random" }},
        {{ label: "No Action", value: selectedRow.no_action, selected: payload.selected_policy === "no_action" }},
      ].map(item => `
        <div class="score-card ${{item.selected ? "active" : ""}}">
          <div class="label">${{item.label}}</div>
          <div class="value ${{item.value > 0 ? "good" : item.value > -2 ? "warn" : "bad"}}">${{fmtSigned(item.value, 3)}}</div>
        </div>
      `).join("");
      const outcomeCards = payload.policy_outcome.map((item) => `
        <div class="runtime-card">
          <div class="label">${{item.label}}</div>
          <div class="value">${{item.value}}</div>
        </div>
      `).join("");
      const eventItems = (demo.events && demo.events.length ? demo.events : ["No retry or terminal events emitted in this trace."]).map((event) => `
        <div class="log-item">
          <div class="log-message">${{event}}</div>
        </div>
      `).join("");
      const deltaCards = [
        {{
          label: "HR delta",
          value: `${{fmtSigned((currentFrame.heart_rate_bpm ?? 0) - (previousFrame.heart_rate_bpm ?? currentFrame.heart_rate_bpm), 1)}} bpm`,
        }},
        {{
          label: "SpO2 delta",
          value: `${{fmtSigned(((currentFrame.spo2 ?? 0) - (previousFrame.spo2 ?? currentFrame.spo2)) * 100, 1)}} pts`,
        }},
        {{
          label: "Blood volume delta",
          value: `${{fmtSigned((currentFrame.blood_volume_ml ?? 0) - (previousFrame.blood_volume_ml ?? currentFrame.blood_volume_ml), 0)}} mL`,
        }},
        {{
          label: "Step reward",
          value: fmtSigned(currentFrame.reward ?? 0, 3),
        }},
      ].map((item) => `
        <div class="runtime-card">
          <div class="label">${{item.label}}</div>
          <div class="value">${{item.value}}</div>
        </div>
      `).join("");
      const logItems = demo.action_log.map((item, idx) => `
        <div class="log-item ${{idx + 1 === frameIndex ? "active" : ""}}">
          <div class="log-head">
            <div class="log-tool">${{idx + 1}}. ${{item.tool_name}}</div>
            <div class="log-reward">${{fmtSigned(item.reward, 3)}}</div>
          </div>
          <div class="log-message">${{item.message}}</div>
        </div>
      `).join("");
      const benchmarkRows = payload.benchmark_matrix.map((row) => `
        <tr>
          <td><strong>${{row.label}}</strong></td>
          <td class="${{row.expert > 0 ? "good" : "bad"}}">${{fmtSigned(row.expert, 3)}}</td>
          <td class="${{row.llm_demo > 0 ? "good" : "bad"}}">${{fmtSigned(row.llm_demo, 3)}}</td>
          <td class="${{row.random > 0 ? "good" : "bad"}}">${{fmtSigned(row.random, 3)}}</td>
          <td class="${{row.no_action > 0 ? "good" : "bad"}}">${{fmtSigned(row.no_action, 3)}}</td>
        </tr>
      `).join("");
      const toolGroups = payload.tool_surface.groups.map((group) => `
        <div class="tool-card">
          <div class="tool-card-head">
            <div class="tool-card-title">${{group.name}}</div>
            <div class="tool-card-count">${{group.tools.length}} tools</div>
          </div>
          <div class="tool-card-copy">${{group.summary}}</div>
          <div class="tool-chip-grid">${{group.tools.map((tool) => `<span class="tool-chip">${{tool}}</span>`).join("")}}</div>
        </div>
      `).join("");
      const layerCards = payload.engine_layers.map((layer) => `
        <div class="layer-card">
          <div class="layer-title">${{layer.title}}</div>
          <div class="layer-value">${{layer.value}}</div>
          <div class="layer-detail">${{layer.detail}}</div>
        </div>
      `).join("");
      const runtimeCards = payload.runtime_profile.map((item) => `
        <div class="runtime-card">
          <div class="label">${{item.label}}</div>
          <div class="value">${{item.value}}</div>
        </div>
      `).join("");
      const frameSummary = `
        <div class="frame-card">
          <div class="frame-tool">${{currentFrame.tool_name}}</div>
          <div class="frame-message">${{currentFrame.message}}</div>
          <div class="frame-meta">
            <span class="pill">step ${{currentFrame.step_index}}</span>
            <span class="pill">reward ${{fmtSigned(currentFrame.reward, 3)}}</span>
            <span class="pill">time ${{Math.round(currentFrame.sim_time_s)}}s</span>
            <span class="pill">${{currentFrame.active_alerts.length}} alerts</span>
          </div>
        </div>
      `;

      root.innerHTML = `
        <section class="hero">
          <div class="panel hero-main">
            <div class="eyebrow">Pulse-ER <span>OpenEnv Hackathon</span></div>
            <h1 class="hero-title">${{payload.hero.title}}</h1>
            <div class="hero-subtitle">${{payload.hero.subtitle}}</div>
            <p class="hero-copy">${{payload.hero.description}}</p>
            <div class="badge-row">${{payload.hero.badges.map((badge) => `<span class="badge">${{badge}}</span>`).join("")}}</div>
          </div>
          <div class="panel hero-side">
            <span class="section-label">Engine Stack</span>
            <div class="layer-grid">${{layerCards}}</div>
          </div>
        </section>

        <section class="main-grid">
          <div class="stack">
            <div class="panel section">
              <div class="section-header">
                <h2 class="section-title">Scenario Console</h2>
                <div class="section-meta">${{payload.selected_scenario.replace(/_/g, " ")}}</div>
              </div>
              <div class="scenario-head">
                <span class="section-label">${{payload.scenario.tag}}</span>
                <h3 class="scenario-title">${{payload.scenario.title}}</h3>
                <p class="scenario-copy">${{payload.scenario.summary}}</p>
                <div class="scenario-selector">${{scenarioButtons}}</div>
                <div class="section-meta" style="margin-top:12px;">Policy workbench</div>
                <div class="scenario-selector">${{policyButtons}}</div>
                <div class="microcopy" style="margin-top:12px;">${{selectedPolicyMeta.summary}}</div>
              </div>
              <div class="scenario-contrast">
                <div class="contrast-card bad">
                  <strong>No-treatment baseline</strong>
                  <div>${{payload.scenario.naive_outcome}}</div>
                </div>
                <div class="contrast-card good">
                  <strong>Expert benchmark</strong>
                  <div>${{payload.scenario.trained_outcome}}</div>
                </div>
              </div>
              <div class="microcopy">${{payload.scenario.selected_outcome}}</div>
              <div class="teaching"><strong>Teaching point:</strong> ${{payload.scenario.teaching_point}}</div>
              <div class="score-strip" style="margin-top:18px;">${{scoreStrip}}</div>

              <div class="section-header" style="margin-top:22px;">
                <h3 class="section-title">Patient State Inspector</h3>
                <div class="section-meta">${{payload.selected_policy}} policy trace</div>
              </div>
              <div class="monitor-grid">
                <div class="tile">
                  <div class="tile-label"><span>Heart Rate</span><span class="${{metricStatus("hr", currentFrame.heart_rate_bpm)}}">live</span></div>
                  <div class="tile-value ${{metricStatus("hr", currentFrame.heart_rate_bpm)}}">${{formatBpm(currentFrame.heart_rate_bpm)}}</div>
                  <div class="tile-trend">delta ${{fmtSigned((currentFrame.heart_rate_bpm ?? 0) - (previousFrame.heart_rate_bpm ?? currentFrame.heart_rate_bpm), 1)}}</div>
                </div>
                <div class="tile">
                  <div class="tile-label"><span>Blood Pressure</span><span>perfusion</span></div>
                  <div class="tile-value">${{formatBp(currentFrame.systolic_bp_mmhg, currentFrame.diastolic_bp_mmhg)}}</div>
                  <div class="tile-trend">time ${{Math.round(currentFrame.sim_time_s)}} s</div>
                </div>
                <div class="tile">
                  <div class="tile-label"><span>SpO2</span><span class="${{metricStatus("spo2", currentFrame.spo2)}}">oxygenation</span></div>
                  <div class="tile-value ${{metricStatus("spo2", currentFrame.spo2)}}">${{formatSpo2(currentFrame.spo2)}}</div>
                  <div class="tile-trend">delta ${{fmtSigned(((currentFrame.spo2 ?? 0) - (previousFrame.spo2 ?? currentFrame.spo2)) * 100, 1)}} pts</div>
                </div>
                <div class="tile">
                  <div class="tile-label"><span>Respiratory Rate</span><span class="${{metricStatus("rr", currentFrame.respiration_rate_bpm)}}">ventilation</span></div>
                  <div class="tile-value ${{metricStatus("rr", currentFrame.respiration_rate_bpm)}}">${{formatBpm(currentFrame.respiration_rate_bpm)}}</div>
                  <div class="tile-trend">${{currentFrame.active_alerts.length ? currentFrame.active_alerts.join(" / ") : "no active alerts"}}</div>
                </div>
                <div class="tile">
                  <div class="tile-label"><span>Blood Volume</span><span>circulation</span></div>
                  <div class="tile-value">${{Math.round(currentFrame.blood_volume_ml ?? 0)}} mL</div>
                  <div class="tile-trend">final frame uses current observation volume estimate</div>
                </div>
                <div class="tile">
                  <div class="tile-label"><span>Mental Status</span><span>neuro</span></div>
                  <div class="tile-value">${{currentFrame.mental_status || "n/a"}}</div>
                  <div class="tile-trend">${{currentFrame.active_alerts.length}} active alert(s)</div>
                </div>
              </div>
              <div class="trend-chart">
                ${{buildChart(frames)}}
                <div class="legend">
                  <span class="hr">Heart Rate</span>
                  <span class="spo2">SpO2</span>
                  <span class="reward">Per-step reward</span>
                </div>
                <div class="scrubber">
                  <div class="scrubber-head">
                    <span>Frame scrubber</span>
                    <span>frame ${{currentFrameIndex}} / ${{frames.length - 1}}</span>
                  </div>
                  <input id="frameScrubber" type="range" min="0" max="${{frames.length - 1}}" step="1" value="${{currentFrameIndex}}" />
                </div>
              </div>
              <div class="footer-actions">
                <button class="btn primary" id="runDemoBtn">Run Replay</button>
                <a class="btn secondary" href="${{payload.links.training_url}}" target="_blank" rel="noreferrer">View Training Code</a>
                <a class="btn secondary" href="${{payload.links.repo_url}}" target="_blank" rel="noreferrer">GitHub Repo</a>
              </div>
              <div class="microcopy">
                This console is deliberately built around inspectable state rather than a single animation: you can switch scenarios and policies, scrub frames, inspect step deltas, and cross-check any trace against the benchmark matrix below.
              </div>
            </div>

            <div class="panel section">
              <div class="section-header">
                <h3 class="section-title">Tool Contract Explorer</h3>
                <div class="section-meta">${{payload.tool_surface.public_contract_count}} public / ${{payload.tool_surface.clinical_surface_count}} clinical / ${{payload.tool_surface.runtime_name_count}} runtime</div>
              </div>
              <div class="tool-grid">${{toolGroups}}</div>
            </div>

            <div class="panel section">
              <div class="section-header">
                <h3 class="section-title">Benchmark Matrix</h3>
                <div class="section-meta">verified mock policy separation</div>
              </div>
              <div class="matrix-wrapper">
                <table>
                  <thead>
                    <tr>
                      <th>Scenario</th>
                      <th>Expert</th>
                      <th>LLM Demo</th>
                      <th>Random</th>
                      <th>No Action</th>
                    </tr>
                  </thead>
                  <tbody>${{benchmarkRows}}</tbody>
                </table>
              </div>
            </div>
          </div>

          <div class="stack">
            <div class="panel section">
              <div class="section-header">
                <h3 class="section-title">Step Inspector</h3>
                <div class="section-meta">frame ${{currentFrameIndex}}</div>
              </div>
              ${{frameSummary}}
              <div class="runtime-grid" style="margin-top:16px;">${{deltaCards}}</div>
              <div class="log-list" style="margin-top:16px;">${{logItems}}</div>
            </div>

            <div class="panel section">
              <div class="section-header">
                <h3 class="section-title">Policy Outcome</h3>
                <div class="section-meta">${{payload.available_policies.find(item => item.id === payload.selected_policy).label}}</div>
              </div>
              <div class="runtime-grid">${{outcomeCards}}</div>
              <div class="log-list" style="margin-top:16px;">${{eventItems}}</div>
            </div>

            <div class="panel section">
              <div class="section-header">
                <h3 class="section-title">Runtime Profile</h3>
                <div class="section-meta">selected replay settings</div>
              </div>
              <div class="runtime-grid">${{runtimeCards}}</div>
              <div class="microcopy">
                The Space uses the deterministic mock backend for responsiveness, but this panel exposes the same control dimensions the training and evaluation stack cares about: seed, noise, time pressure, and episode budget.
              </div>
            </div>

            <div class="panel section">
              <div class="section-header">
                <h3 class="section-title">Environment Depth</h3>
                <div class="section-meta">what this Space is surfacing</div>
              </div>
              <div class="highlights" style="margin-top:0;">
                ${{
                  payload.research_highlights.map((item) => `
                    <div class="highlight">
                      <div class="highlight-value">${{item.label}}</div>
                      <div class="highlight-label">${{item.value}}</div>
                    </div>
                  `).join("")
                }}
              </div>
              <div class="microcopy">
                This is not just a replay shell. It exposes the public tool contract, the deeper clinical/runtime surfaces behind it, scenario-conditioned policy ranking, and the runtime settings that shape observed behavior.
              </div>
            </div>
          </div>
        </section>
      `;

      const runButton = document.getElementById("runDemoBtn");
      if (runButton) {{
        runButton.onclick = () => {{
          let index = 0;
          runButton.disabled = true;
          runButton.textContent = "Replaying...";
          const tick = () => {{
            render(index);
            index += 1;
            if (index < frames.length) {{
              window.setTimeout(tick, 850);
              return;
            }}
            runButton.disabled = false;
            runButton.textContent = "Run Replay";
          }};
          tick();
        }};
      }}

      const scrubber = document.getElementById("frameScrubber");
      if (scrubber) {{
        scrubber.oninput = (event) => {{
          const value = Number(event.target.value || 0);
          render(value);
        }};
      }}

      document.querySelectorAll("[data-scenario-id]").forEach((button) => {{
        button.onclick = async () => {{
          const scenarioId = button.getAttribute("data-scenario-id");
          if (!scenarioId || scenarioId === payload.selected_scenario) {{
            return;
          }}
          try {{
            await loadDashboard(scenarioId, payload.selected_policy);
          }} catch (error) {{
            console.error(error);
          }}
        }};
      }});

      document.querySelectorAll("[data-policy-id]").forEach((button) => {{
        button.onclick = async () => {{
          const policyId = button.getAttribute("data-policy-id");
          if (!policyId || policyId === payload.selected_policy) {{
            return;
          }}
          try {{
            await loadDashboard(payload.selected_scenario, policyId);
          }} catch (error) {{
            console.error(error);
          }}
        }};
      }});
    }}

    render(0);
  </script>
</body>
</html>"""
