"""Reward engine for Pulse-ER with dense shaping, terminal scoring, and anti-exploitation guards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pulse_physiology_env.models import PulsePhysiologyAction
from pulse_physiology_env.patient_state import PatientState

from .scenarios import ScenarioDefinition


@dataclass
class ActionRecord:
    """One action attempt in the reward engine history."""

    tool_name: str
    arguments: dict[str, Any]
    sim_time_s: float
    success: bool
    tags: tuple[str, ...]


@dataclass
class RewardTracker:
    """Per-episode reward bookkeeping."""

    reward_profile: str
    action_budget_remaining: int
    last_tool_name: str | None = None
    same_tool_called_consecutively: int = 0
    steps_since_last_diagnostic_review: int = 0
    diagnostics_ordered: set[str] = field(default_factory=set)
    action_history: list[ActionRecord] = field(default_factory=list)
    time_to_stabilize_s: float | None = None


@dataclass
class RewardBreakdown:
    """Structured dense and terminal reward components for one step."""

    dense_total: float = 0.0
    terminal_total: float = 0.0
    total: float = 0.0

    r_map_stability: float = 0.0
    r_spo2_efficiency: float = 0.0
    r_lactate_trend: float = 0.0
    r_intervention_safety: float = 0.0
    r_diagnostic_timeliness: float = 0.0
    r_anti_exploitation: float = 0.0

    survival_bonus: float = 0.0
    time_efficiency_bonus: float = 0.0
    sequence_quality_bonus: float = 0.0
    difficulty_multiplier: float = 1.0

    reward_profile: str = "polytrauma"
    action_budget_remaining: int = 0
    same_tool_called_consecutively: int = 0
    steps_since_last_diagnostic_review: int = 0
    terminal_applied: bool = False

    def as_metadata(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "dense_total": self.dense_total,
            "terminal_total": self.terminal_total,
            "r_map_stability": self.r_map_stability,
            "r_spo2_efficiency": self.r_spo2_efficiency,
            "r_lactate_trend": self.r_lactate_trend,
            "r_intervention_safety": self.r_intervention_safety,
            "r_diagnostic_timeliness": self.r_diagnostic_timeliness,
            "r_anti_exploitation": self.r_anti_exploitation,
            "survival_bonus": self.survival_bonus,
            "time_efficiency_bonus": self.time_efficiency_bonus,
            "sequence_quality_bonus": self.sequence_quality_bonus,
            "difficulty_multiplier": self.difficulty_multiplier,
            "reward_profile": self.reward_profile,
            "action_budget_remaining": self.action_budget_remaining,
            "same_tool_called_consecutively": self.same_tool_called_consecutively,
            "steps_since_last_diagnostic_review": self.steps_since_last_diagnostic_review,
            "terminal_applied": self.terminal_applied,
        }


class RewardEngine:
    """Implements the Pulse-ER reward design with dense, terminal, and sequence-aware signals."""

    MAP_TARGET = 65.0
    HYPOXIA_THRESHOLD = 0.90
    SPO2_TARGET = 0.94
    DEFAULT_ACTION_BUDGET = 30
    DIAGNOSTIC_ORDER_WINDOW_S = 180.0
    DIAGNOSTIC_NEGLECT_WINDOW_S = 300.0
    READY_DIAGNOSTIC_GRACE_STEPS = 5

    DIAGNOSTIC_TOOLS = frozenset({"get_blood_gas", "get_cbc", "get_bmp"})
    RESPIRATORY_ASSESSMENT_TOOLS = frozenset({"get_respiratory_status", "summarize_state", "check_deterioration"})
    CARDIAC_ASSESSMENT_TOOLS = frozenset({"get_vitals", "summarize_state", "check_deterioration"})
    BLEEDING_CONTROL_TOOLS = frozenset({"control_bleeding"})
    PRESSOR_TOOLS = frozenset({"give_pressor"})
    DECOMPRESSION_TOOLS = frozenset({"needle_decompression"})
    PERICARDIOCENTESIS_TOOLS = frozenset({"pericardiocentesis"})
    BLOOD_PRODUCTS = frozenset({"blood", "packed_rbc", "packed_rbcs", "prbc", "prbcs"})
    CRYSTALLOIDS = frozenset({"saline"})
    DIFFICULTY_MULTIPLIER = {"easy": 1.0, "medium": 1.5, "hard": 2.0}

    SCENARIO_MILESTONES: dict[str, list[tuple[str, float]]] = {
        "tension_pneumothorax": [
            ("respiratory_assessment", 0.3),
            ("needle_decompression", 0.4),
            ("crystalloid_after_decomp", 0.2),
            ("pressor_if_needed", 0.1),
        ],
        "hemorrhagic_shock": [
            ("bleeding_control", 0.4),
            ("crystalloid_bolus", 0.3),
            ("pressor_if_needed", 0.2),
            ("blood_transfusion_if_severe", 0.1),
        ],
        "cardiac_tamponade": [
            ("cardiac_assessment", 0.3),
            ("pericardiocentesis", 0.5),
            ("crystalloid_bolus", 0.2),
        ],
        "polytrauma": [
            ("respiratory_assessment", 0.15),
            ("needle_decompression", 0.25),
            ("bleeding_control", 0.25),
            ("blood_transfusion_if_severe", 0.2),
            ("pressor_if_needed", 0.15),
        ],
    }

    def start_episode(self, scenario: ScenarioDefinition, initial_state: PatientState) -> RewardTracker:
        action_budget = max(10, int(round(scenario.max_time_s / 60.0)))
        tracker = RewardTracker(
            reward_profile=scenario.reward_profile,
            action_budget_remaining=action_budget,
        )
        if self._is_stabilized(initial_state):
            tracker.time_to_stabilize_s = initial_state.sim_time_s
        return tracker

    def score_step(
        self,
        tracker: RewardTracker,
        *,
        scenario: ScenarioDefinition,
        before: PatientState,
        after: PatientState,
        action: PulsePhysiologyAction,
        success: bool,
        had_error: bool,
    ) -> RewardBreakdown:
        tool_name = action.tool_name.strip()
        arguments = dict(action.arguments)

        self._update_tracker(tracker, before, after, tool_name, arguments, success)
        breakdown = RewardBreakdown(
            reward_profile=tracker.reward_profile,
            difficulty_multiplier=self.DIFFICULTY_MULTIPLIER[scenario.difficulty],
            action_budget_remaining=tracker.action_budget_remaining,
            same_tool_called_consecutively=tracker.same_tool_called_consecutively,
            steps_since_last_diagnostic_review=tracker.steps_since_last_diagnostic_review,
        )

        breakdown.r_map_stability = self._reward_map_stability(before, after)
        breakdown.r_spo2_efficiency = self._reward_spo2_efficiency(before, after)
        breakdown.r_lactate_trend = self._reward_lactate_trend(after)
        breakdown.r_intervention_safety = self._reward_intervention_safety(tracker, before, after, tool_name, arguments)
        breakdown.r_diagnostic_timeliness = self._reward_diagnostic_timeliness(tracker, before, after, tool_name)
        breakdown.r_anti_exploitation = self._reward_anti_exploitation(tracker, after, success, had_error)

        breakdown.dense_total = (
            0.35 * breakdown.r_map_stability
            + 0.25 * breakdown.r_spo2_efficiency
            + 0.20 * breakdown.r_lactate_trend
            + 0.10 * breakdown.r_intervention_safety
            + 0.10 * breakdown.r_diagnostic_timeliness
            + breakdown.r_anti_exploitation
        )

        if after.done:
            breakdown.terminal_applied = True
            breakdown.survival_bonus = 5.0 if self._is_alive(after) else -5.0
            breakdown.time_efficiency_bonus = self._time_efficiency_bonus(tracker, scenario, after)
            breakdown.sequence_quality_bonus = self.evaluate_milestone_sequence(
                tracker.action_history,
                tracker.reward_profile,
            )
            breakdown.terminal_total = (
                breakdown.survival_bonus
                + breakdown.time_efficiency_bonus
                + breakdown.sequence_quality_bonus
            ) * breakdown.difficulty_multiplier

        breakdown.total = float(max(-30.0, min(30.0, breakdown.dense_total + breakdown.terminal_total)))
        return breakdown

    def evaluate_milestone_sequence(
        self,
        action_history: list[ActionRecord],
        reward_profile: str,
    ) -> float:
        milestones = self.SCENARIO_MILESTONES.get(reward_profile, self.SCENARIO_MILESTONES["polytrauma"])
        score = 0.0
        last_idx = -1

        for milestone_name, weight in milestones:
            idx = self._find_milestone_index(action_history, milestone_name)
            if idx > last_idx:
                score += weight
                last_idx = idx
            elif idx != -1:
                score += weight * 0.3
        return score * 2.0

    def _update_tracker(
        self,
        tracker: RewardTracker,
        before: PatientState,
        after: PatientState,
        tool_name: str,
        arguments: dict[str, Any],
        success: bool,
    ) -> None:
        if tracker.last_tool_name == tool_name:
            tracker.same_tool_called_consecutively += 1
        else:
            tracker.last_tool_name = tool_name
            tracker.same_tool_called_consecutively = 1

        tracker.action_budget_remaining = max(0, tracker.action_budget_remaining - 1)

        diagnostic_review = tool_name in self.DIAGNOSTIC_TOOLS and tool_name in before.ready_diagnostics
        if diagnostic_review:
            tracker.steps_since_last_diagnostic_review = 0
        elif after.ready_diagnostics:
            tracker.steps_since_last_diagnostic_review += 1
        else:
            tracker.steps_since_last_diagnostic_review = 0

        if tool_name in self.DIAGNOSTIC_TOOLS and tool_name not in before.pending_diagnostics and tool_name not in before.ready_diagnostics:
            tracker.diagnostics_ordered.add(tool_name)

        tags = self._extract_action_tags(tracker.action_history, tool_name, arguments, success)
        tracker.action_history.append(
            ActionRecord(
                tool_name=tool_name,
                arguments=arguments,
                sim_time_s=after.sim_time_s,
                success=success,
                tags=tags,
            )
        )

        if tracker.time_to_stabilize_s is None and self._is_stabilized(after):
            tracker.time_to_stabilize_s = after.sim_time_s

    def _reward_map_stability(self, before: PatientState, after: PatientState) -> float:
        previous_map = before.mean_arterial_pressure_mmhg or 0.0
        current_map = after.mean_arterial_pressure_mmhg or 0.0
        reward = self._clip((current_map - previous_map) / self.MAP_TARGET, -1.0, 1.0)
        if previous_map < self.MAP_TARGET <= current_map:
            reward += 0.3
        return reward

    def _reward_spo2_efficiency(self, before: PatientState, after: PatientState) -> float:
        previous_spo2 = before.spo2 or 0.0
        current_spo2 = after.spo2 or 0.0
        reward = (current_spo2 - previous_spo2) * 10.0
        if current_spo2 < self.HYPOXIA_THRESHOLD:
            reward -= 0.2
        if previous_spo2 < self.SPO2_TARGET <= current_spo2:
            reward += 0.2
        return self._clip(reward, -1.5, 1.5)

    @staticmethod
    def _reward_lactate_trend(after: PatientState) -> float:
        if after.lactate_trend == "worsening":
            return -1.0
        if after.lactate_trend == "improving":
            return 0.2
        return 0.0

    def _reward_intervention_safety(
        self,
        tracker: RewardTracker,
        before: PatientState,
        after: PatientState,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> float:
        reward = 0.0

        if tool_name == "give_fluids":
            fluid_type = self._normalize_fluid_type(arguments)
            if self._has_pneumothorax_signs(before) and not self._history_has_tag(tracker.action_history, "needle_decompression"):
                reward -= 0.8
            if fluid_type in self.BLOOD_PRODUCTS and self._is_severe_bleed(before):
                reward += 0.1

        if tool_name == "give_pressor" and self._is_volume_depleted(before):
            reward -= 0.5

        if tool_name == "needle_decompression" and not self._has_pneumothorax_signs(before):
            reward -= 0.4

        if tool_name == "pericardiocentesis":
            if "possible_cardiac_tamponade" in before.active_alerts or "active_pericardial_effusion" in before.active_alerts:
                reward += 0.2
            else:
                reward -= 0.6

        if tool_name == "control_bleeding" and before.active_hemorrhages:
            reward += 0.15

        if tool_name == "airway_support" and before.spo2 is not None and before.spo2 >= 0.97 and before.mental_status == "alert":
            reward -= 0.1

        if tool_name == "give_pressor" and after.mean_arterial_pressure_mmhg is not None and after.mean_arterial_pressure_mmhg >= self.MAP_TARGET:
            reward += 0.1

        return self._clip(reward, -1.0, 1.0)

    def _reward_diagnostic_timeliness(
        self,
        tracker: RewardTracker,
        before: PatientState,
        after: PatientState,
        tool_name: str,
    ) -> float:
        reward = 0.0

        is_new_order = (
            tool_name in self.DIAGNOSTIC_TOOLS
            and tool_name not in before.pending_diagnostics
            and tool_name not in before.ready_diagnostics
        )
        if is_new_order and before.sim_time_s < self.DIAGNOSTIC_ORDER_WINDOW_S:
            reward += 0.15
        elif is_new_order and before.sim_time_s > self.DIAGNOSTIC_NEGLECT_WINDOW_S:
            reward -= 0.05

        is_review = tool_name in self.DIAGNOSTIC_TOOLS and tool_name in before.ready_diagnostics
        if is_review:
            reward += 0.05

        if (
            after.sim_time_s > self.DIAGNOSTIC_NEGLECT_WINDOW_S
            and not after.pending_diagnostics
            and not after.ready_diagnostics
            and not tracker.diagnostics_ordered
        ):
            reward -= 0.1

        return self._clip(reward, -0.3, 0.3)

    def _reward_anti_exploitation(
        self,
        tracker: RewardTracker,
        after: PatientState,
        success: bool,
        had_error: bool,
    ) -> float:
        reward = 0.0

        if tracker.same_tool_called_consecutively >= 3:
            reward -= 0.1 * (tracker.same_tool_called_consecutively - 2)

        if tracker.action_budget_remaining < 5 and not self._is_stabilized(after):
            reward -= 0.2

        if after.ready_diagnostics and tracker.steps_since_last_diagnostic_review > self.READY_DIAGNOSTIC_GRACE_STEPS:
            reward -= 0.05

        if had_error:
            reward -= 0.75
        elif not success:
            reward -= 0.25

        return reward

    def _time_efficiency_bonus(
        self,
        tracker: RewardTracker,
        scenario: ScenarioDefinition,
        after: PatientState,
    ) -> float:
        time_to_stabilize = tracker.time_to_stabilize_s
        if time_to_stabilize is None and self._is_stabilized(after):
            time_to_stabilize = after.sim_time_s
        if time_to_stabilize is None:
            return 0.0
        return max(0.0, (scenario.max_time_s - time_to_stabilize) / scenario.max_time_s) * 2.0

    def _find_milestone_index(self, action_history: list[ActionRecord], milestone_name: str) -> int:
        for idx, record in enumerate(action_history):
            if not record.success:
                continue
            if milestone_name in record.tags:
                return idx
        return -1

    def _extract_action_tags(
        self,
        action_history: list[ActionRecord],
        tool_name: str,
        arguments: dict[str, Any],
        success: bool,
    ) -> tuple[str, ...]:
        tags: set[str] = {tool_name}

        if tool_name in self.RESPIRATORY_ASSESSMENT_TOOLS:
            tags.add("respiratory_assessment")
        if tool_name in self.CARDIAC_ASSESSMENT_TOOLS:
            tags.add("cardiac_assessment")
        if tool_name in self.DECOMPRESSION_TOOLS:
            tags.add("needle_decompression")
        if tool_name in self.BLEEDING_CONTROL_TOOLS:
            tags.add("bleeding_control")
        if tool_name in self.PERICARDIOCENTESIS_TOOLS:
            tags.add("pericardiocentesis")

        if tool_name == "give_pressor":
            tags.add("pressor_if_needed")
            pressor = str(arguments.get("pressor") or arguments.get("agent") or "").strip().lower().replace("-", "_")
            if pressor == "norepinephrine":
                tags.add("norepinephrine")

        if tool_name == "give_fluids":
            fluid_type = self._normalize_fluid_type(arguments)
            if fluid_type in self.CRYSTALLOIDS:
                tags.add("crystalloid_bolus")
            if fluid_type in self.BLOOD_PRODUCTS:
                tags.add("blood_transfusion_if_severe")
            if self._history_has_tag(action_history, "needle_decompression"):
                tags.add("crystalloid_after_decomp")

        return tuple(sorted(tags)) if success else tuple(sorted({tool_name}))

    def _history_has_tag(self, action_history: list[ActionRecord], tag: str) -> bool:
        return any(record.success and tag in record.tags for record in action_history)

    @staticmethod
    def _normalize_fluid_type(arguments: dict[str, Any]) -> str:
        return str(arguments.get("fluid_type") or arguments.get("fluid") or "saline").strip().lower().replace("-", "_")

    @staticmethod
    def _has_pneumothorax_signs(state: PatientState) -> bool:
        return any(
            alert in state.active_alerts
            for alert in ("possible_tension_pneumothorax", "unilateral_absent_breath_sounds", "bilateral_absent_breath_sounds")
        )

    @staticmethod
    def _is_severe_bleed(state: PatientState) -> bool:
        return bool(state.active_hemorrhages) and sum(state.active_hemorrhages.values()) >= 120.0

    @staticmethod
    def _is_volume_depleted(state: PatientState) -> bool:
        if state.blood_volume_ml is not None and state.blood_volume_ml < 3000.0:
            return True
        if state.shock_index is not None and state.shock_index > 1.2 and state.active_hemorrhages:
            return True
        return False

    @classmethod
    def _is_alive(cls, state: PatientState) -> bool:
        if "cardiac_arrest" in state.active_alerts:
            return False
        if state.mean_arterial_pressure_mmhg is not None and state.mean_arterial_pressure_mmhg <= 10.0:
            return False
        if state.heart_rate_bpm is not None and state.heart_rate_bpm <= 0.1:
            return False
        return True

    @classmethod
    def _is_stabilized(cls, state: PatientState) -> bool:
        if not cls._is_alive(state):
            return False
        if state.mean_arterial_pressure_mmhg is None or state.mean_arterial_pressure_mmhg < cls.MAP_TARGET:
            return False
        if state.spo2 is None or state.spo2 < cls.SPO2_TARGET:
            return False
        if state.mental_status not in {"alert", "verbal"}:
            return False
        if state.lactate_trend == "worsening":
            return False
        blocking_alerts = {
            "active_hemorrhage",
            "possible_tension_pneumothorax",
            "possible_cardiac_tamponade",
            "cardiac_arrest",
        }
        return not any(alert in blocking_alerts for alert in state.active_alerts)

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return max(lower, min(value, upper))
