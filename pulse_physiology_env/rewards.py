"""Reward logic for the mock Pulse-ER environment."""

from __future__ import annotations

from dataclasses import dataclass

from .models import PatientState


READ_ONLY_TOOLS = {
    "get_vitals",
    "summarize_state",
    "check_deterioration",
    "recommend_next_step",
}


@dataclass(frozen=True)
class RewardBreakdown:
    """Interpretable reward components for one step transition."""

    oxygenation_delta: float
    perfusion_delta: float
    heart_rate_stabilization: float
    blood_volume_delta: float
    assessment_bonus: float
    timely_intervention_bonus: float
    deterioration_penalty: float
    collapse_penalty: float
    total: float


def compute_reward(
    previous_state: PatientState,
    new_state: PatientState,
    tool_name: str,
    recommended_actions: tuple[str, ...] = (),
) -> RewardBreakdown:
    """Score one transition in a simple, readable way."""

    oxygenation_delta = (new_state.spo2 - previous_state.spo2) * 20.0
    perfusion_delta = (new_state.systolic_bp_mmhg - previous_state.systolic_bp_mmhg) / 10.0
    heart_rate_stabilization = (previous_state.heart_rate_bpm - new_state.heart_rate_bpm) / 10.0

    blood_volume_delta = 0.0
    if new_state.blood_volume_ml is not None and previous_state.blood_volume_ml is not None:
        blood_volume_delta = (new_state.blood_volume_ml - previous_state.blood_volume_ml) / 250.0

    assessment_bonus = 0.1 if tool_name in READ_ONLY_TOOLS else 0.0
    timely_intervention_bonus = 0.3 if tool_name in recommended_actions else 0.0

    new_alerts = set(new_state.active_alerts)
    previous_alerts = set(previous_state.active_alerts)
    deterioration_penalty = -0.25 * max(0, len(new_alerts) - len(previous_alerts))

    if tool_name == "advance_time" and previous_state.active_alerts:
        deterioration_penalty -= 0.2

    collapse_penalty = 0.0
    if new_state.done and "cardiovascular_collapse" in new_alerts:
        collapse_penalty = -5.0

    total = round(
        oxygenation_delta
        + perfusion_delta
        + heart_rate_stabilization
        + blood_volume_delta
        + assessment_bonus
        + timely_intervention_bonus
        + deterioration_penalty
        + collapse_penalty,
        3,
    )

    return RewardBreakdown(
        oxygenation_delta=round(oxygenation_delta, 3),
        perfusion_delta=round(perfusion_delta, 3),
        heart_rate_stabilization=round(heart_rate_stabilization, 3),
        blood_volume_delta=round(blood_volume_delta, 3),
        assessment_bonus=round(assessment_bonus, 3),
        timely_intervention_bonus=round(timely_intervention_bonus, 3),
        deterioration_penalty=round(deterioration_penalty, 3),
        collapse_penalty=round(collapse_penalty, 3),
        total=total,
    )
