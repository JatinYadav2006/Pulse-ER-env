"""Tier 3 consumer-side workflows for Pulse-ER.

These workflows sit above the low-level tool layer and transform the current
observation or episode trace into judge-friendly and agent-friendly outputs.
They are intentionally backend-agnostic so the same logic can be used against
mock and real Pulse runtimes.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .episode_runner import EpisodeTrace
from .models import PulsePhysiologyObservation


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
) -> DeteriorationExplanation:
    """Explain the likely deterioration driver or current stability."""

    alerts = set(observation.active_alerts)
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
    if previous_observation is not None:
        if (
            previous_observation.spo2 is not None
            and observation.spo2 is not None
            and observation.spo2 < previous_observation.spo2
        ):
            supporting_findings.append("Oxygenation is falling over time.")
        if (
            previous_observation.systolic_bp_mmhg is not None
            and observation.systolic_bp_mmhg is not None
            and observation.systolic_bp_mmhg < previous_observation.systolic_bp_mmhg
        ):
            supporting_findings.append("Systolic pressure is trending down.")

    return DeteriorationExplanation(
        scenario_id=observation.scenario_id,
        status="deteriorating" if observation.active_alerts else "stable",
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
