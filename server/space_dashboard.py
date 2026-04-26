"""HF Space dashboard data and HTML for Pulse-ER."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from pulse_physiology_env.episode_runner import EpisodeRunner, EpisodeTrace
from pulse_physiology_env.eval_mock import score_policy, score_random_policy
from pulse_physiology_env.policies import build_expert_policy
from pulse_physiology_env.server.adapters import KNOWN_TOOL_NAMES, MockPulseAdapter
from pulse_physiology_env.server.pathology_architect import PathologyArchitect
from pulse_physiology_env.demo_llm_policy import heuristic_infer_fn
from pulse_physiology_env.policies import LLMPolicy, build_no_action_policy

DEFAULT_SPACE_SCENARIO = "respiratory_distress"
SPACE_SCENARIO_META: dict[str, dict[str, str]] = {
    "baseline_stable": {
        "label": "Baseline Stable",
        "title": "Baseline Stability Replay",
        "summary": (
            "A low-acuity mock case shown under observation noise and time pressure. "
            "The main signal here is restraint: unnecessary intervention loses reward, while basic assessment keeps the patient stable."
        ),
        "teaching_point": "assess before escalating care",
    },
    "respiratory_distress": {
        "label": "Respiratory Distress",
        "title": "Respiratory Rescue Demo Episode",
        "summary": (
            "A reproducible eight-step respiratory rescue on the deterministic mock backend. "
            "The expert sequence restores oxygenation and lowers respiratory workload under noisy monitoring."
        ),
        "teaching_point": "stabilize oxygenation before waiting",
    },
    "hemorrhagic_shock": {
        "label": "Hemorrhagic Shock",
        "title": "Hemorrhagic Shock Replay",
        "summary": (
            "A circulation-first case where delayed action is punished quickly. "
            "The replay highlights how rapid hemorrhage control and resuscitation separate trained policies from passive ones."
        ),
        "teaching_point": "control shock early and avoid delay",
    },
}


def _patient_count() -> int:
    return len(PathologyArchitect().supported_patients())


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
                "mental_status": getattr(obs.mental_status, "value", obs.mental_status),
                "active_alerts": list(obs.active_alerts),
            }
        )

    final_summary = trace.summary()
    return {
        "scenario_id": trace.scenario_id,
        "policy_name": trace.policy_name,
        "summary": final_summary,
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
        {"label": "LLM Demo", "value": llm_demo.average_reward, "status": "good" if llm_demo.average_reward > 0 else "warn"},
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


@lru_cache(maxsize=8)
def get_demo_episode_payload(scenario_id: str = DEFAULT_SPACE_SCENARIO) -> dict[str, Any]:
    backend = MockPulseAdapter(
        default_scenario_id=scenario_id,
        observation_noise_level=0.3,
        time_pressure_enabled=True,
        seed=0,
    )
    runner = EpisodeRunner(backend=backend, max_steps=8)
    policy = build_expert_policy()
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
            "title": f"{scenario_id.replace('_', ' ').title()} Replay",
            "summary": "Deterministic mock replay for the selected scenario.",
            "teaching_point": "follow the measured physiology signal",
        },
    )
    frames = demo["frames"]
    first_observed = next((frame for frame in frames if frame["spo2"] is not None), frames[0])
    final = demo["summary"]
    expert_rr = benchmarks["per_scenario"]["expert"][scenario_id]
    no_action_rr = benchmarks["per_scenario"]["no_action"][scenario_id]
    return {
        "title": meta["title"],
        "tag": "SPACE REPLAY",
        "summary": meta["summary"],
        "teaching_point": meta["teaching_point"],
        "naive_outcome": (
            f"No-action baseline on {scenario_id}: {no_action_rr:+.3f} reward "
            "with progressive deterioration."
        ),
        "trained_outcome": (
            f"Expert replay improves SpO2 from {first_observed['spo2'] * 100:.1f}% "
            f"to {final['spo2_percent']:.1f}% and finishes at {expert_rr:+.3f} reward."
        ),
    }


def _build_research_highlights(demo: dict[str, Any], benchmarks: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"label": f"{len(KNOWN_TOOL_NAMES)}-tool contract", "value": "public tool surface used by the Space"},
        {"label": f"{_patient_count()} patient profiles", "value": "supported by PathologyArchitect"},
        {"label": f"{benchmarks['comparison'][0]['value']:+.3f} avg", "value": "expert average reward on eval_mock"},
        {"label": f"{benchmarks['comparison'][3]['value']:+.3f} avg", "value": "no-action average reward on eval_mock"},
    ]


def get_dashboard_payload(scenario_id: str | None = None) -> dict[str, Any]:
    benchmarks = get_policy_benchmark_payload()
    selected_scenario = _normalize_space_scenario(scenario_id, benchmarks)
    demo = get_demo_episode_payload(selected_scenario)
    scenario = _build_primary_scenario(selected_scenario, demo, benchmarks)
    return {
        "hero": {
            "title": "Pulse-ER",
            "subtitle": "Emergency Room Dashboard",
            "description": (
                "A trauma-medicine reinforcement-learning environment backed by Pulse physiology. "
                "The dashboard pairs live-looking patient telemetry with measured benchmark evidence "
                "so judges can see both the clinical story and the learning signal immediately."
            ),
            "badges": [f"{len(KNOWN_TOOL_NAMES)}-tool contract", f"{_patient_count()} patient profiles", "Pulse 4.3.2 validated"],
        },
        "selected_scenario": selected_scenario,
        "available_scenarios": [
            {"id": scenario_key, "label": SPACE_SCENARIO_META.get(scenario_key, {}).get("label", scenario_key.replace("_", " ").title())}
            for scenario_key in _available_space_scenarios(benchmarks)
        ],
        "scenario": scenario,
        "policy_comparison": benchmarks["comparison"],
        "research_highlights": _build_research_highlights(demo, benchmarks),
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
  <title>Pulse-ER Emergency Room Dashboard</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: rgba(18, 27, 46, 0.92);
      --panel-strong: rgba(15, 22, 38, 0.98);
      --border: rgba(115, 158, 214, 0.24);
      --text: #eef5ff;
      --muted: #8da2c4;
      --cyan: #57d6ff;
      --teal: #47e5bb;
      --amber: #ffc35a;
      --red: #ff6c81;
      --green: #5cf2a2;
      --shadow: 0 18px 60px rgba(0, 0, 0, 0.35);
      --font: "Segoe UI", "IBM Plex Sans", "Helvetica Neue", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: var(--font);
      background:
        radial-gradient(circle at 15% 20%, rgba(87, 214, 255, 0.18), transparent 28%),
        radial-gradient(circle at 85% 0%, rgba(71, 229, 187, 0.16), transparent 25%),
        linear-gradient(180deg, #09101d 0%, #0b1020 100%);
      color: var(--text);
    }}
    .shell {{
      width: min(1380px, calc(100vw - 32px));
      margin: 24px auto 40px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      border-radius: 24px;
      overflow: hidden;
      backdrop-filter: blur(18px);
    }}
    .hero-main {{
      padding: 28px 30px 26px;
      position: relative;
    }}
    .eyebrow {{
      display: inline-flex;
      gap: 10px;
      align-items: center;
      font-size: 12px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--cyan);
      font-weight: 700;
    }}
    .hero-title {{
      margin: 12px 0 8px;
      font-size: clamp(34px, 5vw, 58px);
      line-height: 0.94;
      letter-spacing: -0.04em;
    }}
    .hero-subtitle {{
      margin: 0 0 14px;
      font-size: clamp(18px, 2vw, 24px);
      color: #d8e7ff;
      font-weight: 600;
    }}
    .hero-copy {{
      margin: 0;
      max-width: 60ch;
      color: var(--muted);
      line-height: 1.65;
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
      background: rgba(90, 115, 169, 0.15);
      border: 1px solid rgba(109, 149, 220, 0.22);
      color: #e7f1ff;
      font-size: 13px;
      font-weight: 600;
    }}
    .hero-side {{
      padding: 24px;
      display: grid;
      gap: 14px;
      align-content: start;
    }}
    .scenario-pill {{
      align-self: start;
      display: inline-flex;
      padding: 7px 12px;
      border-radius: 999px;
      background: rgba(255, 195, 90, 0.12);
      color: var(--amber);
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      font-weight: 800;
    }}
    .scenario-title {{
      margin: 6px 0 8px;
      font-size: 24px;
      line-height: 1.15;
    }}
    .scenario-copy {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
      font-size: 14px;
    }}
    .scenario-selector {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 4px;
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
      gap: 10px;
      margin-top: 6px;
    }}
    .contrast-card {{
      padding: 14px 16px;
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
    .dashboard {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
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
    .monitor-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .tile {{
      padding: 16px 18px;
      border-radius: 18px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
      min-height: 96px;
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
    .status-good {{ color: var(--green); }}
    .status-warn {{ color: var(--amber); }}
    .status-bad {{ color: var(--red); }}
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
    .log-list {{
      display: grid;
      gap: 10px;
      margin-top: 2px;
      max-height: 332px;
      overflow: auto;
      padding-right: 6px;
    }}
    .log-item {{
      padding: 14px 14px 12px;
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
    .benchmarks {{
      display: grid;
      gap: 14px;
    }}
    .benchmark-row {{
      display: grid;
      grid-template-columns: 110px 1fr 64px;
      gap: 12px;
      align-items: center;
    }}
    .benchmark-label {{
      font-size: 13px;
      font-weight: 600;
      color: #dbe7fb;
    }}
    .bar-track {{
      position: relative;
      height: 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.06);
      overflow: hidden;
    }}
    .bar-fill {{
      position: absolute;
      inset: 0 auto 0 0;
      border-radius: inherit;
      width: 0%;
      background: linear-gradient(90deg, var(--cyan), var(--teal));
    }}
    .bar-fill.bad {{
      background: linear-gradient(90deg, #9a5cff, var(--red));
    }}
    .benchmark-value {{
      text-align: right;
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }}
    .highlights {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .highlight {{
      padding: 16px;
      border-radius: 18px;
      background: rgba(8, 14, 27, 0.76);
      border: 1px solid rgba(255,255,255,0.05);
    }}
    .highlight-value {{
      font-size: 22px;
      font-weight: 700;
      letter-spacing: -0.03em;
      margin-bottom: 4px;
    }}
    .highlight-label {{
      font-size: 13px;
      color: var(--muted);
      line-height: 1.4;
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
      transition: transform 160ms ease, opacity 160ms ease;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
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
    @media (max-width: 1100px) {{
      .hero, .dashboard {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 720px) {{
      .shell {{ width: min(100vw - 18px, 100%); margin: 10px auto 26px; }}
      .hero-main, .hero-side, .section {{ padding: 18px; }}
      .monitor-grid, .highlights {{ grid-template-columns: 1fr; }}
      .benchmark-row {{ grid-template-columns: 92px 1fr 56px; }}
      .tile-value {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <div class="shell" id="app"></div>
  <script>
    const initialPayload = {payload};
    let state = initialPayload;

    function formatDelta(value, digits = 1) {{
      if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
      const prefix = value > 0 ? "+" : "";
      return `${{prefix}}${{value.toFixed(digits)}}`;
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
      if (value === null || value === undefined) return "status-warn";
      if (kind === "spo2") return value < 0.9 ? "status-bad" : value < 0.95 ? "status-warn" : "status-good";
      if (kind === "hr") return value > 120 ? "status-bad" : value > 100 ? "status-warn" : "status-good";
      if (kind === "rr") return value > 28 ? "status-bad" : value > 20 ? "status-warn" : "status-good";
      return "status-good";
    }}

    function buildChart(frames) {{
      const width = 620;
      const height = 220;
      const pad = 20;
      const innerWidth = width - pad * 2;
      const innerHeight = height - pad * 2;
      const domainX = frames.map((f, idx) => idx);
      const hrValues = frames.map(f => f.heart_rate_bpm ?? 0);
      const spo2Values = frames.map(f => (f.spo2 ?? 0) * 100);
      const rewardValues = frames.slice(1).map(f => f.reward ?? 0);

      const xMax = Math.max(1, domainX[domainX.length - 1] || 1);
      const yMin = Math.min(...hrValues, ...spo2Values, ...rewardValues, 0);
      const yMax = Math.max(...hrValues, ...spo2Values, ...rewardValues, 120);

      const toX = (index) => pad + (index / xMax) * innerWidth;
      const toY = (value) => pad + innerHeight - ((value - yMin) / (yMax - yMin || 1)) * innerHeight;
      const poly = (values) => values.map((value, idx) => `${{toX(idx)}},${{toY(value)}}`).join(" ");

      return `
        <svg class="chart-svg" viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Episode telemetry chart">
          <rect x="0" y="0" width="${{width}}" height="${{height}}" rx="18" fill="transparent"></rect>
          <line x1="${{pad}}" y1="${{height - pad}}" x2="${{width - pad}}" y2="${{height - pad}}" stroke="rgba(255,255,255,0.12)" />
          <line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height - pad}}" stroke="rgba(255,255,255,0.12)" />
          <polyline fill="none" stroke="#57d6ff" stroke-width="3" points="${{poly(hrValues)}}" />
          <polyline fill="none" stroke="#47e5bb" stroke-width="3" points="${{poly(spo2Values)}}" />
          <polyline fill="none" stroke="#ffc35a" stroke-width="2.5" stroke-dasharray="8 7" points="${{poly([0, ...rewardValues])}}" />
        </svg>
      `;
    }}

    async function loadScenario(scenarioId) {{
      const response = await fetch(`/space/api/dashboard?scenario_id=${{encodeURIComponent(scenarioId)}}`);
      if (!response.ok) {{
        throw new Error(`Failed to load scenario: ${{scenarioId}}`);
      }}
      state = await response.json();
      render(0);
    }}

    function render(frameIndex = 0) {{
      const root = document.getElementById("app");
      const payload = state;
      const demo = payload.demo_episode;
      const frames = demo.frames;
      const currentFrame = frames[Math.min(frameIndex, frames.length - 1)];
      const previousFrame = frames[Math.max(0, Math.min(frameIndex - 1, frames.length - 1))];
      const policyRows = payload.policy_comparison.map((item) => {{
        const normalized = Math.min(100, Math.max(6, ((item.value + 13) / 17) * 100));
        return `
          <div class="benchmark-row">
            <div class="benchmark-label">${{item.label}}</div>
            <div class="bar-track"><div class="bar-fill ${{item.status === "bad" ? "bad" : ""}}" style="width:${{normalized}}%"></div></div>
            <div class="benchmark-value">${{item.value > 0 ? "+" : ""}}${{item.value.toFixed(2)}}</div>
          </div>
        `;
      }}).join("");

      const logItems = demo.action_log.map((item, idx) => `
        <div class="log-item ${{idx + 1 === frameIndex ? "active" : ""}}">
          <div class="log-head">
            <div class="log-tool">${{idx + 1}}. ${{item.tool_name}}</div>
            <div class="log-reward">${{item.reward > 0 ? "+" : ""}}${{item.reward.toFixed(3)}}</div>
          </div>
          <div class="log-message">${{item.message}}</div>
        </div>
      `).join("");

      const highlights = payload.research_highlights.map((item) => `
        <div class="highlight">
          <div class="highlight-value">${{item.label}}</div>
          <div class="highlight-label">${{item.value}}</div>
        </div>
      `).join("");
      const scenarioButtons = payload.available_scenarios.map((item) => `
        <button class="scenario-chip ${{item.id === payload.selected_scenario ? "active" : ""}}" data-scenario-id="${{item.id}}">
          ${{item.label}}
        </button>
      `).join("");

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
            <span class="scenario-pill">${{payload.scenario.tag}}</span>
            <h2 class="scenario-title">${{payload.scenario.title}}</h2>
            <p class="scenario-copy">${{payload.scenario.summary}}</p>
            <div class="scenario-selector">${{scenarioButtons}}</div>
            <div class="scenario-contrast">
              <div class="contrast-card bad">
                <strong>Naive Path</strong>
                <div>${{payload.scenario.naive_outcome}}</div>
              </div>
              <div class="contrast-card good">
                <strong>Trained Path</strong>
                <div>${{payload.scenario.trained_outcome}}</div>
              </div>
            </div>
            <p class="scenario-copy"><strong style="color:var(--amber);text-transform:uppercase;font-size:11px;letter-spacing:0.14em;">Teaching Point</strong><br>${{payload.scenario.teaching_point}}</p>
          </div>
        </section>

        <section class="dashboard">
          <div class="stack">
            <div class="panel section">
              <div class="section-header">
                <h3 class="section-title">Patient Monitor</h3>
                <div class="section-meta">${{demo.scenario_id.replace(/_/g, " ")}}</div>
              </div>
              <div class="monitor-grid">
                <div class="tile">
                  <div class="tile-label"><span>Heart Rate</span><span class="${{metricStatus("hr", currentFrame.heart_rate_bpm)}}">live</span></div>
                  <div class="tile-value ${{metricStatus("hr", currentFrame.heart_rate_bpm)}}">${{formatBpm(currentFrame.heart_rate_bpm)}}</div>
                  <div class="tile-trend">delta ${{formatDelta((currentFrame.heart_rate_bpm ?? 0) - (previousFrame.heart_rate_bpm ?? currentFrame.heart_rate_bpm), 1)}}</div>
                </div>
                <div class="tile">
                  <div class="tile-label"><span>Blood Pressure</span><span class="status-good">perfusion</span></div>
                  <div class="tile-value">${{formatBp(currentFrame.systolic_bp_mmhg, currentFrame.diastolic_bp_mmhg)}}</div>
                  <div class="tile-trend">time ${{Math.round(currentFrame.sim_time_s)}} s</div>
                </div>
                <div class="tile">
                  <div class="tile-label"><span>SpO2</span><span class="${{metricStatus("spo2", currentFrame.spo2)}}">oxygenation</span></div>
                  <div class="tile-value ${{metricStatus("spo2", currentFrame.spo2)}}">${{formatSpo2(currentFrame.spo2)}}</div>
                  <div class="tile-trend">delta ${{formatDelta(((currentFrame.spo2 ?? 0) - (previousFrame.spo2 ?? currentFrame.spo2)) * 100, 1)}} pts</div>
                </div>
                <div class="tile">
                  <div class="tile-label"><span>Respiratory Rate</span><span class="${{metricStatus("rr", currentFrame.respiration_rate_bpm)}}">ventilation</span></div>
                  <div class="tile-value ${{metricStatus("rr", currentFrame.respiration_rate_bpm)}}">${{formatBpm(currentFrame.respiration_rate_bpm)}}</div>
                  <div class="tile-trend">${{currentFrame.active_alerts.length ? currentFrame.active_alerts.join(" / ") : "no active alerts"}}</div>
                </div>
              </div>
              <div class="trend-chart">
                ${{buildChart(frames)}}
                <div class="legend">
                  <span class="hr">Heart Rate</span>
                  <span class="spo2">SpO2</span>
                  <span class="reward">Per-step reward</span>
                </div>
              </div>
              <div class="footer-actions">
                <button class="btn primary" id="runDemoBtn">Run Demo Episode</button>
                <a class="btn secondary" href="${{payload.links.training_url}}" target="_blank" rel="noreferrer">View Training Code</a>
                <a class="btn secondary" href="${{payload.links.repo_url}}" target="_blank" rel="noreferrer">GitHub Repo</a>
              </div>
              <div class="microcopy">
                This replay uses the repo's deterministic mock backend with observation noise and time pressure enabled.
                It is fast enough for a Space demo, but still shows the real policy ordering that matters.
              </div>
            </div>

            <div class="panel section">
              <div class="section-header">
                <h3 class="section-title">Agent Action Log</h3>
                <div class="section-meta">${{demo.policy_name}} policy</div>
              </div>
              <div class="log-list">${{logItems}}</div>
            </div>
          </div>

          <div class="stack">
            <div class="panel section">
              <div class="section-header">
                <h3 class="section-title">Benchmark Evidence</h3>
                <div class="section-meta">verified mock ranking</div>
              </div>
              <div class="benchmarks">${{policyRows}}</div>
              <div class="highlights">${{highlights}}</div>
            </div>

            <div class="panel section">
              <div class="section-header">
                <h3 class="section-title">Episode Outcome</h3>
                <div class="section-meta">${{demo.summary.termination_reason}}</div>
              </div>
              <div class="highlights">
                <div class="highlight">
                  <div class="highlight-value">${{demo.summary.total_reward > 0 ? "+" : ""}}${{demo.summary.total_reward}}</div>
                  <div class="highlight-label">total reward on the demo episode</div>
                </div>
                <div class="highlight">
                  <div class="highlight-value">${{demo.summary.spo2_percent}}%</div>
                  <div class="highlight-label">final oxygen saturation after the expert sequence</div>
                </div>
                <div class="highlight">
                  <div class="highlight-value">${{demo.summary.num_steps}}</div>
                  <div class="highlight-label">actions before max-timestep cutoff</div>
                </div>
                <div class="highlight">
                  <div class="highlight-value">${{demo.summary.mental_status}}</div>
                  <div class="highlight-label">final mental status</div>
                </div>
              </div>
              <div class="microcopy">
                The right column is deliberately evidence-heavy: benchmark ordering, adversarial survival findings,
                and a reproducible episode trace. The left column tells the clinical story in monitor form.
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
            runButton.textContent = "Run Demo Episode";
          }};
          tick();
        }};
      }}
      document.querySelectorAll("[data-scenario-id]").forEach((button) => {{
        button.onclick = async () => {{
          const scenarioId = button.getAttribute("data-scenario-id");
          if (!scenarioId || scenarioId === payload.selected_scenario) {{
            return;
          }}
          try {{
            await loadScenario(scenarioId);
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
