"""Reward logic for the mock Pulse-ER environment."""

from __future__ import annotations

from dataclasses import dataclass

from .models import PatientState


READ_ONLY_TOOLS = {
    "get_vitals",
    "get_respiratory_status",
    "get_blood_gas",
    "get_cbc",
    "get_bmp",
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
    anti_exploitation_penalty: float
    deterioration_penalty: float
    time_pressure_penalty: float
    collapse_penalty: float
    total: float


def compute_reward(
    previous_state: PatientState,
    new_state: PatientState,
    tool_name: str,
    recommended_actions: tuple[str, ...] = (),
    *,
    tool_usage_count: int = 1,
    same_tool_called_consecutively: int = 1,
    state_changed: bool = True,
    time_pressure_multiplier: float = 1.0,
) -> RewardBreakdown:
    """Score one transition in a simple, readable way."""

    oxygenation_delta = (new_state.spo2 - previous_state.spo2) * 20.0
    perfusion_delta = (new_state.systolic_bp_mmhg - previous_state.systolic_bp_mmhg) / 10.0
    heart_rate_stabilization = (previous_state.heart_rate_bpm - new_state.heart_rate_bpm) / 10.0

    blood_volume_delta = 0.0
    if new_state.blood_volume_ml is not None and previous_state.blood_volume_ml is not None:
        blood_volume_delta = (new_state.blood_volume_ml - previous_state.blood_volume_ml) / 250.0

    assessment_bonus = 0.1 if tool_name in READ_ONLY_TOOLS and tool_usage_count == 1 else 0.0
    timely_intervention_bonus = 0.3 if tool_name in recommended_actions and tool_usage_count == 1 else 0.0

    anti_exploitation_penalty = 0.0
    if same_tool_called_consecutively >= 2:
        anti_exploitation_penalty -= 0.15 * (same_tool_called_consecutively - 1)
    if tool_name in READ_ONLY_TOOLS and tool_usage_count > 1 and not state_changed:
        anti_exploitation_penalty -= 0.05
    if tool_name not in READ_ONLY_TOOLS and not state_changed:
        anti_exploitation_penalty -= 0.05

    new_alerts = set(new_state.active_alerts)
    previous_alerts = set(previous_state.active_alerts)
    deterioration_penalty = -0.25 * max(0, len(new_alerts) - len(previous_alerts))

    if tool_name == "advance_time" and previous_state.active_alerts:
        deterioration_penalty -= 0.2

    time_pressure_penalty = 0.0
    if time_pressure_multiplier > 1.0 and _is_unstable(new_state):
        time_pressure_penalty -= 0.3 * (time_pressure_multiplier - 1.0)
        if tool_name == "advance_time":
            time_pressure_penalty -= 0.15 * (time_pressure_multiplier - 1.0)

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
        + anti_exploitation_penalty
        + deterioration_penalty
        + time_pressure_penalty
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
        anti_exploitation_penalty=round(anti_exploitation_penalty, 3),
        deterioration_penalty=round(deterioration_penalty, 3),
        time_pressure_penalty=round(time_pressure_penalty, 3),
        collapse_penalty=round(collapse_penalty, 3),
        total=total,
    )


def _is_unstable(state: PatientState) -> bool:
    systolic = state.systolic_bp_mmhg if state.systolic_bp_mmhg is not None else 120.0
    spo2 = state.spo2 if state.spo2 is not None else 1.0
    return bool(state.active_alerts) or systolic < 95.0 or spo2 < 0.92 or state.mental_status != "alert"
