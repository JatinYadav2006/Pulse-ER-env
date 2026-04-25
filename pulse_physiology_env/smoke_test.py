"""Smoke tests for deterministic mock scenarios.

Run from the repo root with:

    python -m pulse_physiology_env.smoke_test
"""

from __future__ import annotations

from dataclasses import dataclass

from pulse_physiology_env.models import ToolAction
from pulse_physiology_env.server.adapters import MockPulseAdapter


@dataclass(frozen=True)
class ScenarioRun:
    """Defines one deterministic trajectory to validate."""

    scenario_id: str
    label: str
    actions: tuple[ToolAction, ...]


def action(tool_name: str, **arguments) -> ToolAction:
    """Small helper for compact trajectory definitions."""

    return ToolAction(tool_name=tool_name, arguments=arguments)


GOOD_RUNS = (
    ScenarioRun(
        scenario_id="baseline_stable",
        label="good",
        actions=(
            action("get_vitals"),
            action("check_deterioration"),
            action("advance_time", seconds=30),
            action("summarize_state"),
        ),
    ),
    ScenarioRun(
        scenario_id="respiratory_distress",
        label="good",
        actions=(
            action("get_vitals"),
            action("give_oxygen", flow_lpm=15),
            action("position_patient"),
            action("airway_support"),
            action("advance_time", seconds=30),
            action("give_oxygen", flow_lpm=15),
            action("airway_support", mode="basic"),
            action("advance_time", seconds=30),
        ),
    ),
    ScenarioRun(
        scenario_id="hemorrhagic_shock",
        label="good",
        actions=(
            action("get_vitals"),
            action("control_bleeding"),
            action("give_fluids", volume_ml=500),
            action("give_oxygen", flow_lpm=15),
            action("position_patient", position="supine"),
            action("give_fluids", volume_ml=250),
            action("check_deterioration"),
            action("advance_time", seconds=30),
        ),
    ),
)

BAD_RUNS = (
    ScenarioRun(
        scenario_id="baseline_stable",
        label="bad",
        actions=(
            action("advance_time", seconds=120),
            action("advance_time", seconds=120),
        ),
    ),
    ScenarioRun(
        scenario_id="respiratory_distress",
        label="bad",
        actions=(
            action("advance_time", seconds=30),
            action("advance_time", seconds=30),
            action("advance_time", seconds=30),
        ),
    ),
    ScenarioRun(
        scenario_id="hemorrhagic_shock",
        label="bad",
        actions=(
            action("advance_time", seconds=30),
            action("advance_time", seconds=30),
            action("advance_time", seconds=30),
        ),
    ),
)


def run_scenario(run: ScenarioRun) -> tuple[float, dict]:
    """Execute one trajectory and capture the final summary."""

    adapter = MockPulseAdapter(default_scenario_id=run.scenario_id)
    result = adapter.reset(run.scenario_id)
    total_reward = result.reward

    for tool_action in run.actions:
        result = adapter.step(tool_action)
        total_reward += result.reward

    state = result.observation
    summary = {
        "scenario_id": run.scenario_id,
        "label": run.label,
        "total_reward": round(total_reward, 3),
        "sim_time_s": state.sim_time_s,
        "heart_rate_bpm": state.heart_rate_bpm,
        "systolic_bp_mmhg": state.systolic_bp_mmhg,
        "diastolic_bp_mmhg": state.diastolic_bp_mmhg,
        "spo2": state.spo2,
        "respiration_rate_bpm": state.respiration_rate_bpm,
        "blood_volume_ml": state.blood_volume_ml,
        "mental_status": state.mental_status,
        "active_alerts": state.active_alerts,
        "done": state.done,
    }
    return total_reward, summary


def print_summary(summary: dict) -> None:
    """Pretty-print a trajectory outcome."""

    print(f"[{summary['scenario_id']}] {summary['label']} run")
    print(f"  reward: {summary['total_reward']:.3f}")
    print(f"  time: {summary['sim_time_s']:.0f}s")
    print(
        "  vitals:"
        f" HR {summary['heart_rate_bpm']:.1f},"
        f" BP {summary['systolic_bp_mmhg']:.1f}/{summary['diastolic_bp_mmhg']:.1f},"
        f" SpO2 {summary['spo2']:.3f},"
        f" RR {summary['respiration_rate_bpm']:.1f}"
    )
    print(f"  blood_volume_ml: {summary['blood_volume_ml']}")
    print(f"  mental_status: {summary['mental_status']}")
    print(f"  alerts: {', '.join(summary['active_alerts']) if summary['active_alerts'] else 'none'}")
    print(f"  done: {summary['done']}")


def main() -> None:
    """Run scenario smoke tests and compare good vs bad trajectories."""

    good_scores: dict[str, float] = {}
    bad_scores: dict[str, float] = {}

    print("Running mock scenario smoke tests...\n")

    for run in GOOD_RUNS:
        score, summary = run_scenario(run)
        good_scores[run.scenario_id] = score
        print_summary(summary)
        print()

    for run in BAD_RUNS:
        score, summary = run_scenario(run)
        bad_scores[run.scenario_id] = score
        print_summary(summary)
        print()

    print("Comparison checks:")
    failures = []
    for scenario_id, good_score in good_scores.items():
        bad_score = bad_scores[scenario_id]
        passed = good_score > bad_score
        status = "PASS" if passed else "FAIL"
        print(
            f"  {status} {scenario_id}:"
            f" good_reward={good_score:.3f}"
            f" bad_reward={bad_score:.3f}"
        )
        if not passed:
            failures.append(scenario_id)

    if failures:
        raise SystemExit(f"Smoke test failed for scenarios: {', '.join(failures)}")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
