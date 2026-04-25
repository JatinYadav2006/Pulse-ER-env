"""Tier 3 consumer-side workflows for Pulse-ER.

These workflows sit above the low-level tool layer and transform the current
observation or episode trace into judge-friendly and agent-friendly outputs.
They are intentionally backend-agnostic so the same logic can be used against
mock and real Pulse runtimes.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from .episode_runner import EpisodeTrace
from .models import PulsePhysiologyObservation


class DeteriorationStatus(str, Enum):
    """Clinical status buckets used for judge-facing Tier 3 reasoning."""

    STABLE = "stable"
    MONITORING = "monitoring"
    DETERIORATING = "deteriorating"
    CRITICAL = "critical"


MINOR_ALERTS = {"tachycardia", "tachypnea"}
MAJOR_ALERTS = {"hypotension", "hypoxemia", "blood_loss", "cardiovascular_collapse"}
MAJOR_VITALS = (
    "mean_arterial_pressure_mmhg",
    "spo2",
    "heart_rate_bpm",
    "respiration_rate_bpm",
)
CRITICAL_ACCELERATION_VITALS = (
    "mean_arterial_pressure_mmhg",
    "spo2",
    "heart_rate_bpm",
)


def _mental_status_value(mental_status) -> str:
    return getattr(mental_status, "value", str(mental_status))


def _risk_level(observation: PulsePhysiologyObservation) -> str:
    alerts = set(observation.active_alerts)
    mental_status = _mental_status_value(observation.mental_status)
    if observation.done or "cardiovascular_collapse" in alerts or mental_status == "unresponsive":
        return "critical"
    if {"hypotension", "blood_loss", "hypoxemia"} & alerts or mental_status in {"pain", "verbal"}:
        return "high"
    if alerts:
        return "moderate"
    return "low"


def _priority_reasons(observation: PulsePhysiologyObservation) -> list[str]:
    alerts = set(observation.active_alerts)
    reasons: list[str] = []
    if "blood_loss" in alerts:
        reasons.append("Ongoing blood loss threatens perfusion and should be controlled early.")
    if "hypotension" in alerts:
        reasons.append("Low blood pressure suggests reduced perfusion and possible shock.")
    if "hypoxemia" in alerts:
        reasons.append("Low oxygen saturation requires respiratory support or oxygen therapy.")
    if "tachypnea" in alerts:
        reasons.append("High respiratory rate suggests respiratory stress or compensation.")
    if "tachycardia" in alerts:
        reasons.append("Persistent tachycardia may reflect compensation, stress, or ongoing instability.")
    if not reasons:
        reasons.append("No active high-priority alerts are present, so reassessment and safe monitoring are appropriate.")
    return reasons


def _mean_arterial_pressure(observation: PulsePhysiologyObservation) -> float | None:
    """Return MAP directly or derive it from systolic and diastolic pressure."""

    if observation.mean_arterial_pressure_mmhg is not None:
        return observation.mean_arterial_pressure_mmhg
    if observation.systolic_bp_mmhg is None or observation.diastolic_bp_mmhg is None:
        return None
    return (observation.systolic_bp_mmhg + (2 * observation.diastolic_bp_mmhg)) / 3


def _vital_value(observation: PulsePhysiologyObservation, vital_name: str) -> float | None:
    """Resolve one vital from an observation, including derived MAP support."""

    if vital_name == "mean_arterial_pressure_mmhg":
        return _mean_arterial_pressure(observation)
    return getattr(observation, vital_name, None)


def _vital_deviation(vital_name: str, value: float | None) -> float:
    """Measure how far a vital has drifted from its safe range for trend scoring."""

    if value is None:
        return 0.0

    if vital_name == "mean_arterial_pressure_mmhg":
        return max(0.0, 65.0 - value)
    if vital_name == "spo2":
        return max(0.0, 0.94 - value)
    if vital_name == "heart_rate_bpm":
        if 60.0 <= value <= 100.0:
            return 0.0
        return min(abs(value - 60.0), abs(value - 100.0))
    if vital_name == "respiration_rate_bpm":
        if 12.0 <= value <= 20.0:
            return 0.0
        return min(abs(value - 12.0), abs(value - 20.0))
    return 0.0


def _trend_tolerance(vital_name: str) -> float:
    """Return the minimum meaningful drift for one vital trend calculation."""

    if vital_name == "mean_arterial_pressure_mmhg":
        return 3.0
    if vital_name == "spo2":
        return 0.02
    if vital_name == "heart_rate_bpm":
        return 5.0
    if vital_name == "respiration_rate_bpm":
        return 2.0
    return 0.0


def _recent_observations(
    observation: PulsePhysiologyObservation,
    previous_observation: PulsePhysiologyObservation | None = None,
    observations: list[PulsePhysiologyObservation] | None = None,
    *,
    window: int = 3,
) -> list[PulsePhysiologyObservation]:
    """Assemble the most recent observation window for trend-aware Tier 3 logic."""

    recent: list[PulsePhysiologyObservation] = []
    if observations:
        recent.extend(observations)
    elif previous_observation is not None:
        recent.extend([previous_observation, observation])
    else:
        recent.append(observation)

    if not recent or recent[-1] != observation:
        recent.append(observation)
    return recent[-window:]


def get_trend(
    vital_name: str,
    observations: list[PulsePhysiologyObservation],
    window: int = 3,
) -> str:
    """Classify a vital as improving, stable, or worsening over recent observations."""

    recent = observations[-window:]
    deviations = [
        _vital_deviation(vital_name, _vital_value(observation, vital_name))
        for observation in recent
    ]
    if len(deviations) < 2:
        return "stable"

    delta = deviations[-1] - deviations[0]
    tolerance = _trend_tolerance(vital_name)
    if delta > tolerance:
        return "worsening"
    if delta < -tolerance:
        return "improving"
    return "stable"


def _is_accelerating(vital_name: str, observations: list[PulsePhysiologyObservation]) -> bool:
    """Detect acceleration when a vital drifts farther out of range step over step."""

    recent = observations[-3:]
    if len(recent) < 3:
        return False

    deviations = [
        _vital_deviation(vital_name, _vital_value(observation, vital_name))
        for observation in recent
    ]
    first_delta = deviations[1] - deviations[0]
    second_delta = deviations[2] - deviations[1]
    tolerance = _trend_tolerance(vital_name)
    return (
        first_delta > 0
        and second_delta > first_delta
        and second_delta > (tolerance / 2)
        and deviations[-1] > (2 * tolerance)
    )


def _hard_threshold_reason(observation: PulsePhysiologyObservation) -> str | None:
    """Return the first critical threshold breach, if one is present."""

    map_value = _mean_arterial_pressure(observation)
    if map_value is not None and map_value < 50.0:
        return "MAP is below 50 mmHg, indicating critical perfusion failure."
    if observation.spo2 is not None and observation.spo2 < 0.85:
        return "SpO2 is below 85%, indicating critical hypoxemia."
    if observation.heart_rate_bpm is not None and observation.heart_rate_bpm > 150.0:
        return "Heart rate is above 150 bpm, indicating critical cardiovascular stress."
    if observation.heart_rate_bpm is not None and observation.heart_rate_bpm < 40.0:
        return "Heart rate is below 40 bpm, indicating critical bradycardia."
    return None


def _classify_deterioration_status(
    observation: PulsePhysiologyObservation,
    recent_observations: list[PulsePhysiologyObservation],
) -> tuple[DeteriorationStatus, str, dict[str, str]]:
    """Classify status from alert severity and recent vital trends."""

    alerts = set(observation.active_alerts)
    trend_map = {
        vital_name: get_trend(vital_name, recent_observations)
        for vital_name in MAJOR_VITALS
    }

    hard_threshold_reason = _hard_threshold_reason(observation)
    if hard_threshold_reason is not None:
        return DeteriorationStatus.CRITICAL, hard_threshold_reason, trend_map

    if any(_is_accelerating(vital_name, recent_observations) for vital_name in CRITICAL_ACCELERATION_VITALS):
        return (
            DeteriorationStatus.CRITICAL,
            "At least one major vital is accelerating away from the safe range.",
            trend_map,
        )

    worsening_major_vitals = [
        vital_name
        for vital_name, trend in trend_map.items()
        if vital_name in {"mean_arterial_pressure_mmhg", "spo2", "heart_rate_bpm"} and trend == "worsening"
    ]
    minor_alert_count = len(alerts & MINOR_ALERTS)

    if not alerts and all(trend != "worsening" for trend in trend_map.values()):
        return DeteriorationStatus.STABLE, "No active alerts and no worsening vital trends are present.", trend_map

    if worsening_major_vitals or minor_alert_count >= 2:
        return (
            DeteriorationStatus.DETERIORATING,
            "A major vital is trending the wrong way or multiple minor alerts are accumulating.",
            trend_map,
        )

    return (
        DeteriorationStatus.MONITORING,
        "Only minor or non-progressive abnormalities are present, so close monitoring is appropriate.",
        trend_map,
    )


class NextStepRecommendation(BaseModel):
    """Tier 3 recommendation for the next best action."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    risk_level: str
    recommended_tool: str
    arguments: dict = Field(default_factory=dict)
    rationale: str
    alternatives: list[str] = Field(default_factory=list)


class TriageSummary(BaseModel):
    """Tier 3 triage framing for the current patient state."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    acuity: str
    headline: str
    active_alerts: list[str]
    vitals_snapshot: dict
    immediate_focus: list[str]


class DeteriorationExplanation(BaseModel):
    """Tier 3 explanation of why the patient is worsening or stable."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    status: str
    cascade_risk: str
    primary_driver: str
    supporting_findings: list[str]
    recommended_response: str


class InterventionPlan(BaseModel):
    """Tier 3 short intervention plan based on current state."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    risk_level: str
    ordered_steps: list[dict]
    monitoring_targets: list[str]
    escalation_trigger: str


class EpisodeReport(BaseModel):
    """Tier 3 episode-level report for demos and judge summaries."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    policy_name: str
    total_reward: float
    outcome: str
    key_actions: list[str]
    final_alerts: list[str]
    summary: str


def recommend_next_step(observation: PulsePhysiologyObservation) -> NextStepRecommendation:
    """Recommend the next best tool call from the current observation."""

    alerts = set(observation.active_alerts)
    available_tools = set(observation.available_tools)
    risk_level = _risk_level(observation)

    if "blood_loss" in alerts and "control_bleeding" in available_tools:
        return NextStepRecommendation(
            scenario_id=observation.scenario_id,
            risk_level=risk_level,
            recommended_tool="control_bleeding",
            rationale="Active blood loss is present and should be controlled before deterioration progresses.",
            alternatives=[tool for tool in ("give_fluids", "give_oxygen", "check_deterioration") if tool in available_tools],
        )
    if (
        observation.scenario_id == "hemorrhagic_shock"
        and "tachycardia" in alerts
        and "give_fluids" in available_tools
    ):
        return NextStepRecommendation(
            scenario_id=observation.scenario_id,
            risk_level=risk_level,
            recommended_tool="give_fluids",
            arguments={"volume_ml": 250},
            rationale="Persistent tachycardia after initial hemorrhage control suggests the patient may still benefit from additional perfusion support.",
            alternatives=[tool for tool in ("check_deterioration", "summarize_state", "advance_time") if tool in available_tools],
        )
    if "hypotension" in alerts and "give_fluids" in available_tools:
        return NextStepRecommendation(
            scenario_id=observation.scenario_id,
            risk_level=risk_level,
            recommended_tool="give_fluids",
            arguments={"volume_ml": 500},
            rationale="Hypotension suggests poor perfusion and fluid resuscitation is the next most direct support.",
            alternatives=[tool for tool in ("control_bleeding", "position_patient", "check_deterioration") if tool in available_tools],
        )
    if "hypoxemia" in alerts and "give_oxygen" in available_tools:
        return NextStepRecommendation(
            scenario_id=observation.scenario_id,
            risk_level=risk_level,
            recommended_tool="give_oxygen",
            arguments={"flow_lpm": 15},
            rationale="Hypoxemia is active and oxygen support is the fastest way to improve oxygenation.",
            alternatives=[tool for tool in ("airway_support", "position_patient", "check_deterioration") if tool in available_tools],
        )
    if "tachypnea" in alerts and "airway_support" in available_tools:
        return NextStepRecommendation(
            scenario_id=observation.scenario_id,
            risk_level=risk_level,
            recommended_tool="airway_support",
            arguments={"mode": "basic"},
            rationale="Respiratory effort remains elevated and airway support may prevent further deterioration.",
            alternatives=[tool for tool in ("give_oxygen", "position_patient", "check_deterioration") if tool in available_tools],
        )
    if "check_deterioration" in available_tools:
        return NextStepRecommendation(
            scenario_id=observation.scenario_id,
            risk_level=risk_level,
            recommended_tool="check_deterioration",
            rationale="The patient is not in obvious immediate crisis, so reassessment is the safest next step.",
            alternatives=[tool for tool in ("summarize_state", "advance_time") if tool in available_tools],
        )
    return NextStepRecommendation(
        scenario_id=observation.scenario_id,
        risk_level=risk_level,
        recommended_tool="advance_time",
        arguments={"seconds": 30},
        rationale="No higher-priority intervention is exposed, so advance time to generate the next signal.",
        alternatives=[],
    )


def build_triage_summary(observation: PulsePhysiologyObservation) -> TriageSummary:
    """Generate a compact triage summary from the current state."""

    acuity = _risk_level(observation)
    alerts = list(observation.active_alerts)
    mental_status = _mental_status_value(observation.mental_status)
    headline = (
        f"{observation.scenario_id}: {acuity.upper()} acuity with "
        f"HR {observation.heart_rate_bpm:.0f}, "
        f"BP {observation.systolic_bp_mmhg:.0f}/{observation.diastolic_bp_mmhg:.0f}, "
        f"SpO2 {observation.spo2:.2f}, "
        f"mental status {mental_status}."
    )
    return TriageSummary(
        scenario_id=observation.scenario_id,
        acuity=acuity,
        headline=headline,
        active_alerts=alerts,
        vitals_snapshot={
            "heart_rate_bpm": observation.heart_rate_bpm,
            "systolic_bp_mmhg": observation.systolic_bp_mmhg,
            "diastolic_bp_mmhg": observation.diastolic_bp_mmhg,
            "spo2": observation.spo2,
            "respiration_rate_bpm": observation.respiration_rate_bpm,
            "blood_volume_ml": observation.blood_volume_ml,
        },
        immediate_focus=_priority_reasons(observation),
    )


def explain_deterioration(
    observation: PulsePhysiologyObservation,
    previous_observation: PulsePhysiologyObservation | None = None,
    observations: list[PulsePhysiologyObservation] | None = None,
) -> DeteriorationExplanation:
    """Explain the likely deterioration driver or current stability."""

    recent_observations = _recent_observations(
        observation,
        previous_observation,
        observations,
        window=3,
    )
    alerts = set(observation.active_alerts)
    status, status_reason, trend_map = _classify_deterioration_status(observation, recent_observations)

    if "blood_loss" in alerts:
        primary_driver = "hemorrhagic shock physiology"
        response = "Control bleeding and support perfusion with fluids before reassessing."
    elif observation.scenario_id == "hemorrhagic_shock" and "tachycardia" in alerts:
        primary_driver = "residual shock burden after initial resuscitation"
        response = "Reassess perfusion closely and consider additional volume support if the trend does not settle."
    elif "hypoxemia" in alerts:
        primary_driver = "respiratory decompensation"
        response = "Provide oxygen and airway or positioning support, then reassess oxygenation."
    elif "tachycardia" in alerts and "hypotension" in alerts:
        primary_driver = "compensated shock"
        response = "Support perfusion and reassess for ongoing blood loss or inadequate resuscitation."
    elif alerts:
        primary_driver = "ongoing physiological stress"
        response = "Use focused reassessment and the highest-yield intervention exposed by the current tool set."
    else:
        primary_driver = "no active deterioration signal"
        response = "Continue reassessment and controlled monitoring over time."

    supporting_findings = _priority_reasons(observation)
    supporting_findings.append(status_reason)
    if trend_map["spo2"] == "worsening":
        supporting_findings.append("Oxygenation is worsening over the recent observation window.")
    elif trend_map["spo2"] == "improving":
        supporting_findings.append("Oxygenation is improving over the recent observation window.")

    if trend_map["mean_arterial_pressure_mmhg"] == "worsening":
        supporting_findings.append("Perfusion is worsening based on the recent MAP trend.")
    elif trend_map["mean_arterial_pressure_mmhg"] == "improving":
        supporting_findings.append("Perfusion is improving based on the recent MAP trend.")

    if trend_map["heart_rate_bpm"] == "worsening":
        supporting_findings.append("Heart rate is drifting farther from the safe range.")
    if trend_map["respiration_rate_bpm"] == "worsening":
        supporting_findings.append("Respiratory rate is moving in the wrong direction.")

    cascade_risk = "low"
    if status == DeteriorationStatus.DETERIORATING:
        cascade_risk = "medium"
    elif status == DeteriorationStatus.CRITICAL:
        cascade_risk = "imminent"

    return DeteriorationExplanation(
        scenario_id=observation.scenario_id,
        status=status.value,
        cascade_risk=cascade_risk,
        primary_driver=primary_driver,
        supporting_findings=supporting_findings,
        recommended_response=response,
    )


def generate_intervention_plan(observation: PulsePhysiologyObservation) -> InterventionPlan:
    """Create a short ordered plan from the current observation."""

    recommendation = recommend_next_step(observation)
    steps: list[dict] = [
        {
            "priority": 1,
            "tool_name": recommendation.recommended_tool,
            "arguments": recommendation.arguments,
            "why": recommendation.rationale,
        }
    ]
    priority = 2
    for alternative in recommendation.alternatives[:3]:
        steps.append(
            {
                "priority": priority,
                "tool_name": alternative,
                "arguments": {},
                "why": f"Keep {alternative} ready if the primary step does not adequately stabilize the patient.",
            }
        )
        priority += 1

    return InterventionPlan(
        scenario_id=observation.scenario_id,
        risk_level=recommendation.risk_level,
        ordered_steps=steps,
        monitoring_targets=[
            "heart_rate_bpm",
            "systolic_bp_mmhg",
            "spo2",
            "respiration_rate_bpm",
            "active_alerts",
        ],
        escalation_trigger="Escalate if alerts increase, mental status worsens, or perfusion/oxygenation declines after intervention.",
    )


def build_episode_report(trace: EpisodeTrace) -> EpisodeReport:
    """Summarize one episode into a compact Tier 3 report."""

    final_alerts = list(trace.final_observation.active_alerts)
    if trace.final_observation.done:
        outcome = "critical deterioration"
    elif final_alerts:
        outcome = "partially stabilized"
    else:
        outcome = "stabilized"

    key_actions: list[str] = []
    for step in trace.steps:
        if step.action.tool_name not in key_actions:
            key_actions.append(step.action.tool_name)

    summary = (
        f"{trace.policy_name} completed {trace.num_steps} steps in {trace.scenario_id} "
        f"with total reward {trace.total_reward:.3f}; outcome: {outcome}."
    )
    return EpisodeReport(
        scenario_id=trace.scenario_id,
        policy_name=trace.policy_name,
        total_reward=trace.total_reward,
        outcome=outcome,
        key_actions=key_actions,
        final_alerts=final_alerts,
        summary=summary,
    )
