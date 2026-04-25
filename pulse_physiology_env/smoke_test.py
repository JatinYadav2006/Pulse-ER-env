"""Smoke tests for deterministic mock scenarios.

Run from the repo root with:

    python -m pulse_physiology_env.smoke_test
"""

from __future__ import annotations

from dataclasses import dataclass

from pulse_physiology_env.models import PulsePhysiologyObservation, ToolAction
from pulse_physiology_env.server.adapters import MockPulseAdapter
from pulse_physiology_env.tier3_workflows import explain_deterioration


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
        f" SpO2 {summary['spo2'] * 100:.1f}%,"
        f" RR {summary['respiration_rate_bpm']:.1f}"
    )
    print(f"  blood_volume_ml: {summary['blood_volume_ml']}")
    print(f"  mental_status: {summary['mental_status']}")
    print(f"  alerts: {', '.join(summary['active_alerts']) if summary['active_alerts'] else 'none'}")
    print(f"  done: {summary['done']}")


def _observation_fixture(
    *,
    sim_time_s: float,
    heart_rate_bpm: float,
    systolic_bp_mmhg: float,
    diastolic_bp_mmhg: float,
    spo2: float,
    respiration_rate_bpm: float,
    active_alerts: list[str],
) -> PulsePhysiologyObservation:
    """Build a compact observation fixture for Tier 3 regression checks."""

    return PulsePhysiologyObservation(
        scenario_id="regression_case",
        patient_id="tier3_fixture",
        sim_time_s=sim_time_s,
        heart_rate_bpm=heart_rate_bpm,
        systolic_bp_mmhg=systolic_bp_mmhg,
        diastolic_bp_mmhg=diastolic_bp_mmhg,
        spo2=spo2,
        respiration_rate_bpm=respiration_rate_bpm,
        blood_volume_ml=5400.0,
        active_alerts=active_alerts,
        available_tools=["get_vitals", "check_deterioration", "advance_time"],
    )


def _regression_check_tier3_statuses() -> None:
    """Validate nuanced Tier 3 status classification across core severity buckets."""

    stable_history = [
        _observation_fixture(
            sim_time_s=0.0,
            heart_rate_bpm=72.0,
            systolic_bp_mmhg=118.0,
            diastolic_bp_mmhg=76.0,
            spo2=0.98,
            respiration_rate_bpm=14.0,
            active_alerts=[],
        ),
        _observation_fixture(
            sim_time_s=30.0,
            heart_rate_bpm=71.0,
            systolic_bp_mmhg=119.0,
            diastolic_bp_mmhg=77.0,
            spo2=0.98,
            respiration_rate_bpm=14.0,
            active_alerts=[],
        ),
        _observation_fixture(
            sim_time_s=60.0,
            heart_rate_bpm=70.0,
            systolic_bp_mmhg=120.0,
            diastolic_bp_mmhg=78.0,
            spo2=0.99,
            respiration_rate_bpm=14.0,
            active_alerts=[],
        ),
    ]
    monitoring_history = [
        _observation_fixture(
            sim_time_s=0.0,
            heart_rate_bpm=102.0,
            systolic_bp_mmhg=118.0,
            diastolic_bp_mmhg=76.0,
            spo2=0.97,
            respiration_rate_bpm=18.0,
            active_alerts=["tachycardia"],
        ),
        _observation_fixture(
            sim_time_s=30.0,
            heart_rate_bpm=104.0,
            systolic_bp_mmhg=118.0,
            diastolic_bp_mmhg=76.0,
            spo2=0.97,
            respiration_rate_bpm=18.0,
            active_alerts=["tachycardia"],
        ),
        _observation_fixture(
            sim_time_s=60.0,
            heart_rate_bpm=106.0,
            systolic_bp_mmhg=117.0,
            diastolic_bp_mmhg=75.0,
            spo2=0.97,
            respiration_rate_bpm=18.0,
            active_alerts=["tachycardia"],
        ),
    ]
    deteriorating_history = [
        _observation_fixture(
            sim_time_s=0.0,
            heart_rate_bpm=108.0,
            systolic_bp_mmhg=110.0,
            diastolic_bp_mmhg=70.0,
            spo2=0.95,
            respiration_rate_bpm=20.0,
            active_alerts=["tachycardia"],
        ),
        _observation_fixture(
            sim_time_s=30.0,
            heart_rate_bpm=118.0,
            systolic_bp_mmhg=96.0,
            diastolic_bp_mmhg=60.0,
            spo2=0.93,
            respiration_rate_bpm=22.0,
            active_alerts=["tachycardia", "hypotension"],
        ),
        _observation_fixture(
            sim_time_s=60.0,
            heart_rate_bpm=128.0,
            systolic_bp_mmhg=85.0,
            diastolic_bp_mmhg=50.0,
            spo2=0.91,
            respiration_rate_bpm=26.0,
            active_alerts=["tachycardia", "hypotension", "tachypnea"],
        ),
    ]
    critical_history = [
        _observation_fixture(
            sim_time_s=0.0,
            heart_rate_bpm=120.0,
            systolic_bp_mmhg=100.0,
            diastolic_bp_mmhg=64.0,
            spo2=0.92,
            respiration_rate_bpm=24.0,
            active_alerts=["hypoxemia", "tachypnea"],
        ),
        _observation_fixture(
            sim_time_s=30.0,
            heart_rate_bpm=132.0,
            systolic_bp_mmhg=94.0,
            diastolic_bp_mmhg=60.0,
            spo2=0.88,
            respiration_rate_bpm=28.0,
            active_alerts=["hypoxemia", "tachypnea"],
        ),
        _observation_fixture(
            sim_time_s=60.0,
            heart_rate_bpm=146.0,
            systolic_bp_mmhg=88.0,
            diastolic_bp_mmhg=56.0,
            spo2=0.82,
            respiration_rate_bpm=32.0,
            active_alerts=["hypoxemia", "tachypnea", "tachycardia"],
        ),
    ]

    expectations = (
        ("stable", "low", stable_history),
        ("monitoring", "low", monitoring_history),
        ("deteriorating", "medium", deteriorating_history),
        ("critical", "imminent", critical_history),
    )
    for expected_status, expected_risk, history in expectations:
        explanation = explain_deterioration(history[-1], observations=history)
        if explanation.status != expected_status or explanation.cascade_risk != expected_risk:
            raise SystemExit(
                "Tier 3 regression failed: expected "
                f"{expected_status}/{expected_risk}, got {explanation.status}/{explanation.cascade_risk}."
            )


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

    _regression_check_tier3_statuses()
    print("PASS Tier 3 status classification")
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
