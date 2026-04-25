"""Policy abstractions and baseline policies for Pulse-ER episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable, Protocol

from .models import EnvironmentResponse, PulsePhysiologyObservation, ToolAction
from .prompt_builder import build_policy_prompt
from .tool_availability import ToolAvailabilityError, validate_tool_availability
from .tool_parser import ToolParseError, parse_tool_action


READ_ONLY_TOOLS = {
    "get_vitals",
    "summarize_state",
    "check_deterioration",
    "get_respiratory_status",
    "get_blood_gas",
    "get_cbc",
    "get_bmp",
}

REAL_TRAUMA_SCENARIOS = {
    "polytrauma_demo",
    "trauma_easy_soldier",
    "trauma_medium_carol",
    "trauma_hard_underweight",
}


def action(tool_name: str, **arguments) -> ToolAction:
    """Small helper for compact action construction."""

    return ToolAction(tool_name=tool_name, arguments=arguments)


def _validated_available_tools(observation: PulsePhysiologyObservation) -> list[str]:
    """Validate backend-exposed tools before any policy consumes them."""

    return validate_tool_availability(observation.available_tools)


EXPERT_PLAYBOOKS: dict[str, tuple[ToolAction, ...]] = {
    "baseline_stable": (
        action("get_vitals"),
        action("check_deterioration"),
        action("advance_time", seconds=30),
        action("summarize_state"),
    ),
    "respiratory_distress": (
        action("get_vitals"),
        action("give_oxygen", flow_lpm=15),
        action("position_patient"),
        action("airway_support"),
        action("advance_time", seconds=30),
        action("give_oxygen", flow_lpm=15),
        action("airway_support", mode="basic"),
        action("advance_time", seconds=30),
    ),
    "hemorrhagic_shock": (
        action("get_vitals"),
        action("control_bleeding"),
        action("give_fluids", volume_ml=500),
        action("give_oxygen", flow_lpm=15),
        action("position_patient", position="supine"),
        action("give_fluids", volume_ml=250),
        action("check_deterioration"),
        action("advance_time", seconds=30),
    ),
    "polytrauma_demo": (
        action("get_vitals"),
        action("check_deterioration"),
        action("give_oxygen", monitor_seconds=60),
        action("needle_decompression", monitor_seconds=90),
        action("give_fluids", volume_ml=500, rate_ml_per_min=100, monitor_seconds=60),
        action("recommend_next_step"),
        action("advance_time", seconds=30),
    ),
    "trauma_easy_soldier": (
        action("get_vitals"),
        action("check_deterioration"),
        action("needle_decompression", monitor_seconds=90),
        action("give_oxygen", monitor_seconds=60),
        action("give_fluids", volume_ml=500, rate_ml_per_min=100, monitor_seconds=60),
        action("recommend_next_step"),
        action("advance_time", seconds=30),
    ),
    "trauma_medium_carol": (
        action("get_vitals"),
        action("check_deterioration"),
        action("needle_decompression", monitor_seconds=90),
        action("give_oxygen", monitor_seconds=60),
        action("give_fluids", volume_ml=500, rate_ml_per_min=100, monitor_seconds=60),
        action("get_blood_gas"),
        action("recommend_next_step"),
        action("advance_time", seconds=30),
    ),
    "trauma_hard_underweight": (
        action("get_vitals"),
        action("check_deterioration"),
        action("needle_decompression", monitor_seconds=90),
        action("give_oxygen", monitor_seconds=60),
        action("give_fluids", volume_ml=500, rate_ml_per_min=100, monitor_seconds=60),
        action("get_blood_gas"),
        action("recommend_next_step"),
        action("advance_time", seconds=30),
    ),
}

NO_ACTION_PLAYBOOKS: dict[str, tuple[ToolAction, ...]] = {
    "baseline_stable": (
        action("advance_time", seconds=120),
        action("advance_time", seconds=120),
    ),
    "respiratory_distress": (
        action("advance_time", seconds=30),
        action("advance_time", seconds=30),
        action("advance_time", seconds=30),
    ),
    "hemorrhagic_shock": (
        action("advance_time", seconds=30),
        action("advance_time", seconds=30),
        action("advance_time", seconds=30),
    ),
    "polytrauma_demo": (
        action("advance_time", seconds=30),
        action("advance_time", seconds=30),
        action("advance_time", seconds=30),
    ),
    "trauma_easy_soldier": (
        action("advance_time", seconds=30),
        action("advance_time", seconds=30),
        action("advance_time", seconds=30),
    ),
    "trauma_medium_carol": (
        action("advance_time", seconds=30),
        action("advance_time", seconds=30),
        action("advance_time", seconds=30),
    ),
    "trauma_hard_underweight": (
        action("advance_time", seconds=30),
        action("advance_time", seconds=30),
        action("advance_time", seconds=30),
    ),
}


class Policy(Protocol):
    """Minimal policy interface for episode runners."""

    name: str

    def reset(self, scenario_id: str) -> None:
        """Prepare the policy for a new episode."""

    def select_action(self, observation: PulsePhysiologyObservation) -> ToolAction:
        """Choose the next action from the current observation."""

    def observe_outcome(self, action: ToolAction, result: EnvironmentResponse) -> None:
        """Receive the result of the last action if the policy tracks memory."""


@dataclass
class ScriptedPolicy:
    """Policy that replays a pre-authored action script."""

    name: str
    playbooks: dict[str, tuple[ToolAction, ...]]
    fallback_action: ToolAction = field(
        default_factory=lambda: action("advance_time", seconds=30)
    )
    _scenario_id: str = field(init=False, default="")
    _cursor: int = field(init=False, default=0)

    def reset(self, scenario_id: str) -> None:
        self._scenario_id = scenario_id
        self._cursor = 0

    def select_action(self, observation: PulsePhysiologyObservation) -> ToolAction:
        script = self.playbooks.get(self._scenario_id, ())
        if self._cursor < len(script):
            next_action = script[self._cursor]
        else:
            next_action = self.fallback_action
        self._cursor += 1
        return next_action.model_copy(deep=True)

    def observe_outcome(self, action: ToolAction, result: EnvironmentResponse) -> None:
        return None


@dataclass
class ExpertPolicy:
    """Expert baseline that uses scripts for mock cases and heuristics for real trauma cases."""

    playbooks: dict[str, tuple[ToolAction, ...]]
    name: str = "expert"
    fallback_action: ToolAction = field(
        default_factory=lambda: action("advance_time", seconds=30)
    )
    _scenario_id: str = field(init=False, default="")
    _cursor: int = field(init=False, default=0)
    _recent_actions: list[dict[str, Any]] = field(init=False, default_factory=list)

    def reset(self, scenario_id: str) -> None:
        """Prepare expert policy state for a new episode."""

        self._scenario_id = scenario_id
        self._cursor = 0
        self._recent_actions = []

    def select_action(self, observation: PulsePhysiologyObservation) -> ToolAction:
        """Choose the next expert action for the current backend state."""

        if self._scenario_id in REAL_TRAUMA_SCENARIOS:
            return self._select_real_trauma_action(observation)

        script = self.playbooks.get(self._scenario_id, ())
        if self._cursor < len(script):
            next_action = script[self._cursor]
        else:
            next_action = self.fallback_action
        self._cursor += 1
        return next_action.model_copy(deep=True)

    def observe_outcome(self, action: ToolAction, result: EnvironmentResponse) -> None:
        """Track recent actions so the real trauma heuristics can avoid unsafe loops."""

        self._recent_actions.append(
            {
                "tool_name": action.tool_name,
                "arguments": dict(action.arguments),
                "error": result.error.model_dump() if result.error else None,
            }
        )
        self._recent_actions = self._recent_actions[-8:]

    def _select_real_trauma_action(self, observation: PulsePhysiologyObservation) -> ToolAction:
        """Choose a state-aware action for the real trauma scenarios.

        The real Pulse runtime exposes multi-problem polytrauma cases where a
        fixed script is too brittle. This policy prioritizes decompression,
        hemorrhage control, resuscitation, and pressors based on the current
        state rather than blindly replaying a canned sequence.
        """

        available_tools = set(_validated_available_tools(observation))

        if not self._recent_actions and "get_vitals" in available_tools:
            return action("get_vitals")

        if self._needs_needle_decompression(observation) and "needle_decompression" in available_tools:
            if not self._recently_used("needle_decompression", window=2):
                return action(
                    "needle_decompression",
                    side=self._suggest_needle_side(observation),
                    monitor_seconds=90,
                )

        bleeding_site = self._highest_flow_hemorrhage(observation)
        if bleeding_site and "control_bleeding" in available_tools:
            if not self._recently_controlled_site(bleeding_site, window=1):
                return action(
                    "control_bleeding",
                    site=bleeding_site,
                    method=self._bleeding_method(bleeding_site),
                    monitor_seconds=60,
                )

        map_value = self._map_value(observation)
        if self._needs_volume_resuscitation(observation, map_value) and "give_fluids" in available_tools:
            if not self._recently_used("give_fluids", window=2):
                return action(
                    "give_fluids",
                    fluid_type=self._preferred_fluid(observation),
                    volume_ml=500,
                    rate_ml_per_min=150,
                    monitor_seconds=60,
                )

        if self._needs_pressor_support(observation, map_value) and "give_pressor" in available_tools:
            if not self._recently_used("give_pressor", window=2):
                return action(
                    "give_pressor",
                    pressor="norepinephrine",
                    rate_ml_per_min=7.8,
                    monitor_seconds=60,
                )

        if self._needs_oxygen(observation) and "give_oxygen" in available_tools:
            if not self._recently_used("give_oxygen", window=2):
                return action(
                    "give_oxygen",
                    device=self._oxygen_device(observation),
                    flow_lpm=15 if (observation.spo2 or 1.0) < 0.92 else 8,
                    monitor_seconds=60,
                )

        if self._needs_airway_support(observation) and "airway_support" in available_tools:
            if not self._recently_used("airway_support", window=2):
                return action(
                    "airway_support",
                    support_type=self._airway_support_mode(observation),
                    monitor_seconds=60,
                )

        if self._needs_position_adjustment(observation) and "position_patient" in available_tools:
            if not self._recently_used("position_patient", window=2):
                return action(
                    "position_patient",
                    position="supine" if (map_value or 0.0) < 65 else "upright",
                )

        for diagnostic_tool in ("get_blood_gas", "get_cbc", "get_bmp"):
            if diagnostic_tool in observation.ready_diagnostics and diagnostic_tool in available_tools:
                if not self._recently_used(diagnostic_tool, window=2):
                    return action(diagnostic_tool)

        if (
            "check_deterioration" in available_tools
            and not observation.active_hemorrhages
            and (map_value is None or map_value >= 65)
            and (observation.spo2 is None or observation.spo2 >= 0.92)
            and not self._recently_used("check_deterioration", window=3)
        ):
            return action("check_deterioration")

        if "recommend_next_step" in available_tools and not self._recently_used("recommend_next_step", window=3):
            return action("recommend_next_step")

        if "get_vitals" in available_tools and not self._recently_used("get_vitals", window=2):
            return action("get_vitals")

        return self.fallback_action.model_copy(deep=True)

    def _recently_used(self, tool_name: str, *, window: int) -> bool:
        """Return whether the tool was used recently enough to warrant a pause."""

        return any(
            item["tool_name"] == tool_name
            for item in self._recent_actions[-window:]
        )

    def _recently_controlled_site(self, site: str, *, window: int) -> bool:
        """Return whether the same hemorrhage site was just targeted."""

        for item in self._recent_actions[-window:]:
            if item["tool_name"] != "control_bleeding":
                continue
            recent_site = str(item["arguments"].get("site") or item["arguments"].get("compartment") or "")
            if recent_site == site:
                return True
        return False

    @staticmethod
    def _map_value(observation: PulsePhysiologyObservation) -> float | None:
        """Compute MAP from explicit or derived blood pressure values."""

        if observation.mean_arterial_pressure_mmhg is not None:
            return observation.mean_arterial_pressure_mmhg
        if observation.systolic_bp_mmhg is None or observation.diastolic_bp_mmhg is None:
            return None
        return (observation.systolic_bp_mmhg + 2 * observation.diastolic_bp_mmhg) / 3.0

    @staticmethod
    def _highest_flow_hemorrhage(observation: PulsePhysiologyObservation) -> str | None:
        """Pick the most urgent active hemorrhage site by current estimated flow."""

        if not observation.active_hemorrhages:
            return None
        return max(observation.active_hemorrhages.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _bleeding_method(site: str) -> str:
        """Choose a simple site-appropriate hemorrhage control method."""

        if site in {"left_arm", "right_arm", "left_leg", "right_leg"}:
            return "tourniquet"
        return "pressure"

    @staticmethod
    def _needs_needle_decompression(observation: PulsePhysiologyObservation) -> bool:
        """Detect likely tension physiology from alerts or unilateral absent breath sounds."""

        if {"possible_tension_pneumothorax", "unilateral_absent_breath_sounds"} & set(observation.active_alerts):
            return True
        return "absent left" in observation.breath_sounds or "absent right" in observation.breath_sounds

    @staticmethod
    def _suggest_needle_side(observation: PulsePhysiologyObservation) -> str:
        """Choose the decompression side from breath-sound asymmetry."""

        if "absent left" in observation.breath_sounds:
            return "left"
        if "absent right" in observation.breath_sounds:
            return "right"
        return "left"

    @staticmethod
    def _preferred_fluid(observation: PulsePhysiologyObservation) -> str:
        """Prefer blood products when hemorrhage is ongoing, otherwise saline."""

        if observation.active_hemorrhages or "blood_loss" in observation.active_alerts or "active_hemorrhage" in observation.active_alerts:
            return "blood"
        return "saline"

    @staticmethod
    def _needs_volume_resuscitation(
        observation: PulsePhysiologyObservation,
        map_value: float | None,
    ) -> bool:
        """Detect shock states that should trigger fluid resuscitation."""

        if observation.active_hemorrhages:
            return True
        if map_value is not None and map_value < 65:
            return True
        return observation.shock_index is not None and observation.shock_index >= 0.9

    @staticmethod
    def _needs_pressor_support(
        observation: PulsePhysiologyObservation,
        map_value: float | None,
    ) -> bool:
        """Detect persistent hypotension after volume support has likely started."""

        if map_value is None or map_value >= 60:
            return False
        return any(
            infusion in observation.active_infusions
            for infusion in ("blood", "saline", "packed_rbc", "packed_rbcs")
        )

    @staticmethod
    def _needs_oxygen(observation: PulsePhysiologyObservation) -> bool:
        """Detect hypoxemia or missing oxygen support after thoracic intervention."""

        if observation.spo2 is None:
            return False
        return observation.spo2 < 0.92

    @staticmethod
    def _oxygen_device(observation: PulsePhysiologyObservation) -> str:
        """Choose a simple oxygen delivery device from current saturation."""

        if observation.spo2 is not None and observation.spo2 < 0.9:
            return "non_rebreather_mask"
        if observation.spo2 is not None and observation.spo2 < 0.95:
            return "simple_mask"
        return "nasal_cannula"

    @staticmethod
    def _needs_airway_support(observation: PulsePhysiologyObservation) -> bool:
        """Escalate to airway support when oxygenation or mental status worsens."""

        if observation.airway_support is not None:
            return False
        if observation.spo2 is not None and observation.spo2 < 0.88:
            return True
        return observation.mental_status in {"pain", "unresponsive"} and observation.spo2 is not None and observation.spo2 < 0.92

    @staticmethod
    def _airway_support_mode(observation: PulsePhysiologyObservation) -> str:
        """Choose a simple airway modality from oxygenation and mental status."""

        if observation.spo2 is not None and observation.spo2 < 0.85:
            return "pressure_control_ventilation" if observation.mental_status in {"pain", "unresponsive"} else "bag_valve_mask"
        if observation.spo2 is not None and observation.spo2 < 0.9:
            return "cpap" if observation.mental_status in {"alert", "verbal"} else "bag_valve_mask"
        return "tracheal" if observation.mental_status in {"pain", "unresponsive"} else "nasopharyngeal"

    @staticmethod
    def _needs_position_adjustment(
        observation: PulsePhysiologyObservation,
    ) -> bool:
        """Use positioning only when it meaningfully supports perfusion or oxygenation."""

        if observation.position not in {"supine", "upright"}:
            return True
        if observation.spo2 is not None and observation.spo2 < 0.9 and observation.position != "upright":
            return True
        map_value = ExpertPolicy._map_value(observation)
        return map_value is not None and map_value < 65 and observation.position != "supine"


@dataclass
class RandomPolicy:
    """Policy that samples a valid but uninformed action queue per episode."""

    seed: int = 0
    name: str = "random"
    fallback_action: ToolAction = field(
        default_factory=lambda: action("advance_time", seconds=30)
    )
    _rng: Random = field(init=False)
    _scenario_id: str = field(init=False, default="")
    _queue: list[ToolAction] = field(init=False, default_factory=list)
    _queue_built: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._rng = Random(self.seed)

    def reset(self, scenario_id: str) -> None:
        self._scenario_id = scenario_id
        self._rng = Random(self.seed)
        self._queue = []
        self._queue_built = False

    def select_action(self, observation: PulsePhysiologyObservation) -> ToolAction:
        if not self._queue_built:
            self._build_queue(observation)
        if self._queue:
            return self._queue.pop(0)
        return self.fallback_action.model_copy(deep=True)

    def observe_outcome(self, action: ToolAction, result: EnvironmentResponse) -> None:
        return None

    def _build_queue(self, observation: PulsePhysiologyObservation) -> None:
        """Build a small random action queue from the backend-exposed tools."""

        supported_tools = [
            tool_name
            for tool_name in _validated_available_tools(observation)
            if tool_name not in READ_ONLY_TOOLS and tool_name != "advance_time"
        ]
        self._rng.shuffle(supported_tools)

        max_actions = min(3, len(supported_tools))
        action_count = self._rng.randint(1, max_actions) if max_actions else 0
        selected_tools = supported_tools[:action_count]

        self._queue = [self._tool_action(tool_name) for tool_name in selected_tools]
        self._queue.append(action("advance_time", seconds=30))
        self._queue_built = True

    def _tool_action(self, tool_name: str) -> ToolAction:
        if tool_name == "give_oxygen":
            return action(tool_name, flow_lpm=self._rng.choice((10, 15)))
        if tool_name == "give_fluids":
            return action(tool_name, volume_ml=self._rng.choice((250, 500)))
        return action(tool_name)


@dataclass
class NoActionPolicy:
    """Policy that mostly advances time and lets the patient evolve untreated."""

    seconds_per_step: int = 30
    name: str = "no_action"

    def reset(self, scenario_id: str) -> None:
        return None

    def select_action(self, observation: PulsePhysiologyObservation) -> ToolAction:
        return action("advance_time", seconds=self.seconds_per_step)

    def observe_outcome(self, action: ToolAction, result: EnvironmentResponse) -> None:
        return None


@dataclass
class LLMPolicy:
    """Prompt-driven policy wrapper for future LLM integration."""

    infer_fn: Callable[[str], str]
    name: str = "llm"
    objective: str | None = None
    history_window: int = 4
    anti_repeat_window: int = 2
    fallback_seconds: int = 30
    _recent_history: list[dict[str, Any]] = field(init=False, default_factory=list)

    def reset(self, scenario_id: str) -> None:
        self._recent_history = []

    def select_action(self, observation: PulsePhysiologyObservation) -> ToolAction:
        available_tools = _validated_available_tools(observation)
        prompt = build_policy_prompt(
            observation,
            available_tools=available_tools,
            objective=self.objective,
            recent_history=self._recent_history,
        )
        try:
            raw_response = self.infer_fn(prompt)
            parsed_action = parse_tool_action(raw_response, allowed_tools=available_tools)
        except ToolAvailabilityError:
            raise
        except ToolParseError as exc:
            return self._fallback_action(
                available_tools,
                observation,
                f"Model output parse failed: {exc}",
            )
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            return self._fallback_action(
                available_tools,
                observation,
                f"Model inference failed: {exc}",
            )
        return self._apply_repeat_guard(parsed_action, available_tools)

    def observe_outcome(self, action: ToolAction, result: EnvironmentResponse) -> None:
        mental_status = getattr(result.observation.mental_status, "value", result.observation.mental_status)
        history_entry = {
            "action": action.model_dump(),
            "reward": result.reward,
            "done": result.done,
            "tool_result": result.tool_result.model_dump() if result.tool_result else None,
            "error": result.error.model_dump() if result.error else None,
            "active_alerts": list(result.observation.active_alerts),
            "mental_status": mental_status,
        }
        self._recent_history.append(history_entry)
        self._recent_history = self._recent_history[-self.history_window :]

    def _fallback_action(
        self,
        available_tools: list[str],
        observation: PulsePhysiologyObservation,
        reason: str,
    ) -> ToolAction:
        available_tool_set = set(available_tools)
        if observation.active_alerts and "check_deterioration" in available_tool_set:
            candidate = action("check_deterioration", reasoning=reason)
        elif "advance_time" in available_tool_set:
            candidate = action("advance_time", seconds=self.fallback_seconds, reasoning=reason)
        elif available_tool_set:
            fallback_tool = sorted(available_tool_set)[0]
            candidate = action(fallback_tool, reasoning=reason)
        else:
            raise ToolAvailabilityError("available_tools validation failed before fallback action selection.")
        return self._apply_repeat_guard(candidate, available_tools)

    def _apply_repeat_guard(
        self,
        parsed_action: ToolAction,
        available_tools: list[str],
    ) -> ToolAction:
        if self.anti_repeat_window <= 0 or len(self._recent_history) < self.anti_repeat_window:
            return parsed_action

        recent_tool_names = [
            item["action"]["tool_name"]
            for item in self._recent_history[-self.anti_repeat_window :]
        ]

        if all(tool_name == parsed_action.tool_name for tool_name in recent_tool_names):
            available_tool_set = set(available_tools)
            if parsed_action.tool_name in READ_ONLY_TOOLS and "advance_time" in available_tool_set:
                return action(
                    "advance_time",
                    seconds=self.fallback_seconds,
                    reasoning=(
                        "Avoid repeating the same read-only tool without new information; "
                        "advance time to generate a fresh signal."
                    ),
                )
            if parsed_action.tool_name == "advance_time" and "check_deterioration" in available_tool_set:
                return action(
                    "check_deterioration",
                    reasoning=(
                        "Avoid repeating advance_time without reassessment; "
                        "check for new deterioration before acting again."
                    ),
                )

        return parsed_action


def build_expert_policy() -> ExpertPolicy:
    """Factory for the deterministic-plus-adaptive expert baseline."""

    return ExpertPolicy(playbooks=EXPERT_PLAYBOOKS)


def build_no_action_policy() -> ScriptedPolicy:
    """Factory for the deterministic no-action baseline."""

    return ScriptedPolicy(name="no_action", playbooks=NO_ACTION_PLAYBOOKS)
