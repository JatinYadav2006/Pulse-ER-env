"""Human-readable ATLS judging for demo and evaluation surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

from pulse_physiology_env.patient_state import PatientState

from .reward_engine import ActionRecord

JudgeStatus = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class ATLSCheck:
    """One judge-facing protocol finding."""

    status: JudgeStatus
    title: str
    detail: str
    points: int


@dataclass(frozen=True)
class ATLSJudgeReport:
    """Structured ATLS scorecard with readable rationale."""

    atls_score: int
    verdict: str
    summary: str
    checks: list[ATLSCheck]
    recommended_next_steps: list[str]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""

        return asdict(self)


class ATLSJudge:
    """Produces judge-friendly reasoning without reusing numeric reward output."""

    RESPIRATORY_ASSESSMENT_TOOLS = frozenset(
        {
            "auscultate_chest",
            "get_respiratory_state",
            "get_respiratory_status",
            "run_triage_assessment",
            "detect_deterioration",
            "check_deterioration",
        }
    )
    DIAGNOSTIC_TOOLS = frozenset(
        {
            "order_arterial_blood_gas",
            "get_blood_gas",
            "order_complete_blood_count",
            "get_cbc",
            "order_basic_metabolic_panel",
            "get_bmp",
            "order_point_of_care_ultrasound",
        }
    )
    DECOMPRESSION_TOOLS = frozenset({"needle_decompression", "perform_needle_decompression"})
    BLEEDING_CONTROL_TOOLS = frozenset({"control_bleeding", "apply_tourniquet", "apply_wound_packing", "apply_direct_pressure"})
    VOLUME_SUPPORT_TOOLS = frozenset({"give_fluids", "administer_crystalloid_bolus", "administer_blood_transfusion", "activate_massive_transfusion_protocol"})
    PRESSOR_TOOLS = frozenset({"give_pressor", "start_norepinephrine_infusion", "start_phenylephrine_infusion", "start_dopamine_infusion", "adjust_infusion_rate"})
    AIRWAY_TOOLS = frozenset({"give_oxygen", "apply_nasal_cannula", "apply_simple_mask", "apply_nonrebreather_mask", "airway_support", "apply_bag_valve_mask", "perform_intubation"})
    RESTRICTED_TOOLS = frozenset({"initiate_hemorrhage", "induce_cardiac_arrest", "apply_pericardial_effusion"})
    CPR_TOOLS = frozenset({"perform_cpr"})

    def evaluate(
        self,
        *,
        state: PatientState,
        action_history: Sequence[ActionRecord],
        reward_profile: str,
    ) -> ATLSJudgeReport:
        checks: list[ATLSCheck] = []

        checks.append(self._judge_primary_assessment(action_history))
        checks.append(self._judge_decompression_sequence(action_history, reward_profile))
        checks.append(self._judge_hemorrhage_control(action_history, reward_profile, state))
        checks.append(self._judge_volume_then_pressor(action_history))
        checks.append(self._judge_diagnostics(action_history))
        checks.append(self._judge_safety_violations(action_history))
        checks.append(self._judge_final_state(state))

        score = max(0, min(100, 100 + sum(check.points for check in checks)))
        verdict = self._build_verdict(score)
        summary = self._build_summary(score, checks)
        next_steps = self._recommend_next_steps(state, action_history)
        return ATLSJudgeReport(
            atls_score=score,
            verdict=verdict,
            summary=summary,
            checks=checks,
            recommended_next_steps=next_steps,
        )

    def _judge_primary_assessment(self, action_history: Sequence[ActionRecord]) -> ATLSCheck:
        idx = self._first_index(action_history, self.RESPIRATORY_ASSESSMENT_TOOLS)
        if idx == 0:
            return ATLSCheck("pass", "Primary assessment", "Airway and breathing were assessed first.", 12)
        if idx == 1:
            return ATLSCheck("pass", "Primary assessment", "Assessment happened within the first two actions.", 8)
        if idx != -1:
            return ATLSCheck("warn", "Primary assessment", f"Assessment was delayed until step {idx + 1}.", -6)
        return ATLSCheck("fail", "Primary assessment", "No explicit airway or breathing assessment was documented.", -16)

    def _judge_decompression_sequence(
        self,
        action_history: Sequence[ActionRecord],
        reward_profile: str,
    ) -> ATLSCheck:
        if reward_profile not in {"polytrauma", "tension_pneumothorax"}:
            return ATLSCheck("pass", "Thoracic decompression sequence", "Scenario does not require chest decompression as a dominant milestone.", 0)

        decompression_idx = self._first_index(action_history, self.DECOMPRESSION_TOOLS)
        fluid_idx = self._first_index(action_history, self.VOLUME_SUPPORT_TOOLS)
        if decompression_idx != -1 and (fluid_idx == -1 or decompression_idx < fluid_idx):
            return ATLSCheck("pass", "Thoracic decompression sequence", "Chest decompression occurred before fluid-heavy resuscitation.", 14)
        if fluid_idx != -1 and decompression_idx != -1:
            return ATLSCheck("warn", "Thoracic decompression sequence", "Decompression happened, but only after fluids had already started.", -10)
        if fluid_idx != -1:
            return ATLSCheck("fail", "Thoracic decompression sequence", "Fluids were given without a prior decompression step.", -18)
        return ATLSCheck("warn", "Thoracic decompression sequence", "No decompression step was recorded yet.", -8)

    def _judge_hemorrhage_control(
        self,
        action_history: Sequence[ActionRecord],
        reward_profile: str,
        state: PatientState,
    ) -> ATLSCheck:
        if reward_profile not in {"polytrauma", "hemorrhagic_shock"} and not state.active_hemorrhages:
            return ATLSCheck("pass", "Hemorrhage control", "No dominant hemorrhage-control milestone was expected here.", 0)

        bleed_idx = self._first_index(action_history, self.BLEEDING_CONTROL_TOOLS)
        if bleed_idx == -1:
            return ATLSCheck("fail", "Hemorrhage control", "No tourniquet, packing, direct pressure, or bleed-control action was recorded.", -18)
        if bleed_idx <= 2:
            return ATLSCheck("pass", "Hemorrhage control", "Bleeding control happened early in the resuscitation sequence.", 12)
        return ATLSCheck("warn", "Hemorrhage control", f"Hemorrhage control was delayed until step {bleed_idx + 1}.", -8)

    def _judge_volume_then_pressor(self, action_history: Sequence[ActionRecord]) -> ATLSCheck:
        pressor_idx = self._first_index(action_history, self.PRESSOR_TOOLS)
        if pressor_idx == -1:
            return ATLSCheck("pass", "Pressor sequencing", "No pressor was used before volume status was clarified.", 0)

        volume_idx = self._first_index(action_history, self.VOLUME_SUPPORT_TOOLS)
        if volume_idx != -1 and volume_idx < pressor_idx:
            return ATLSCheck("pass", "Pressor sequencing", "Volume support preceded vasopressor escalation.", 10)
        return ATLSCheck("fail", "Pressor sequencing", "Pressors were started before adequate volume support.", -15)

    def _judge_diagnostics(self, action_history: Sequence[ActionRecord]) -> ATLSCheck:
        diagnostic_idx = self._first_index(action_history, self.DIAGNOSTIC_TOOLS)
        if diagnostic_idx == -1:
            return ATLSCheck("warn", "Diagnostics", "No labs or ultrasound were ordered during the episode.", -6)
        if diagnostic_idx <= 3:
            return ATLSCheck("pass", "Diagnostics", "Diagnostics were ordered early enough to inform management.", 6)
        return ATLSCheck("warn", "Diagnostics", f"Diagnostics did not appear until step {diagnostic_idx + 1}.", -4)

    def _judge_safety_violations(self, action_history: Sequence[ActionRecord]) -> ATLSCheck:
        restricted_idx = self._first_index(action_history, self.RESTRICTED_TOOLS)
        if restricted_idx != -1:
            tool_name = action_history[restricted_idx].tool_name
            return ATLSCheck("fail", "Simulation safety", f"Scenario-authoring tool '{tool_name}' was used inside live care flow.", -20)

        cpr_idx = self._first_index(action_history, self.CPR_TOOLS)
        arrest_idx = self._first_index(action_history, {"induce_cardiac_arrest"})
        if cpr_idx != -1 and (arrest_idx == -1 or arrest_idx > cpr_idx):
            return ATLSCheck("fail", "Simulation safety", "CPR was started without a documented arrest transition.", -16)

        succ_idx = self._first_index(action_history, {"administer_succinylcholine_bolus"})
        airway_idx = self._first_index(action_history, {"perform_intubation"})
        if succ_idx != -1 and (airway_idx == -1 or airway_idx > succ_idx):
            return ATLSCheck("fail", "Simulation safety", "Paralysis was attempted before the airway was secured.", -20)

        return ATLSCheck("pass", "Simulation safety", "No major protocol-level safety violation was detected in the action log.", 6)

    def _judge_final_state(self, state: PatientState) -> ATLSCheck:
        map_ok = state.mean_arterial_pressure_mmhg is not None and state.mean_arterial_pressure_mmhg >= 65.0
        spo2_ok = state.spo2 is not None and state.spo2 >= 0.94
        mental_ok = state.mental_status in {"alert", "verbal"}

        if map_ok and spo2_ok and mental_ok:
            return ATLSCheck("pass", "Physiologic endpoint", "The patient is perfusing and oxygenating at an acceptable endpoint.", 14)
        if map_ok or spo2_ok:
            return ATLSCheck("warn", "Physiologic endpoint", "Resuscitation improved some major endpoints, but not all of them.", -4)
        return ATLSCheck("fail", "Physiologic endpoint", "The patient remains unstable at the end of the sequence.", -14)

    @staticmethod
    def _first_index(action_history: Sequence[ActionRecord], tool_names: set[str] | frozenset[str]) -> int:
        for idx, record in enumerate(action_history):
            if record.success and record.tool_name in tool_names:
                return idx
        return -1

    @staticmethod
    def _build_verdict(score: int) -> str:
        if score >= 85:
            return "ATLS-aligned"
        if score >= 70:
            return "Mostly aligned with timing gaps"
        if score >= 50:
            return "Mixed adherence with visible deviations"
        return "Protocol deviations likely to concern judges"

    @staticmethod
    def _build_summary(score: int, checks: Sequence[ATLSCheck]) -> str:
        failures = [check.title for check in checks if check.status == "fail"]
        if not failures:
            return f"ATLS score {score}/100 with no hard protocol failures."
        joined = ", ".join(failures[:3])
        return f"ATLS score {score}/100. Main concerns: {joined}."

    def _recommend_next_steps(
        self,
        state: PatientState,
        action_history: Sequence[ActionRecord],
    ) -> list[str]:
        recommendations: list[str] = []
        if "possible_tension_pneumothorax" in state.active_alerts and self._first_index(action_history, self.DECOMPRESSION_TOOLS) == -1:
            recommendations.append("Perform needle decompression before more fluids.")
        if state.active_hemorrhages and self._first_index(action_history, self.BLEEDING_CONTROL_TOOLS) == -1:
            recommendations.append("Control hemorrhage immediately with tourniquet or direct pressure.")
        if state.mean_arterial_pressure_mmhg is not None and state.mean_arterial_pressure_mmhg < 65.0 and self._first_index(action_history, self.VOLUME_SUPPORT_TOOLS) == -1:
            recommendations.append("Start volume resuscitation before escalating pressors.")
        if state.spo2 is not None and state.spo2 < 0.92 and self._first_index(action_history, self.AIRWAY_TOOLS) == -1:
            recommendations.append("Escalate oxygenation or airway support now.")
        if not recommendations:
            recommendations.append("Continue reassessment and trend response to the last intervention.")
        return recommendations[:3]
