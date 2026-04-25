"""Policy abstractions and baseline policies for Pulse-ER episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable, Protocol

from .models import EnvironmentResponse, PulsePhysiologyObservation, ToolAction
from .prompt_builder import build_policy_prompt
from .server.mock_scenarios import MOCK_SCENARIOS
from .tool_parser import ToolParseError, parse_tool_action


READ_ONLY_TOOLS = {"get_vitals", "summarize_state", "check_deterioration"}


def action(tool_name: str, **arguments) -> ToolAction:
    """Small helper for compact action construction."""

    return ToolAction(tool_name=tool_name, arguments=arguments)


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

    def __post_init__(self) -> None:
        self._rng = Random(self.seed)

    def reset(self, scenario_id: str) -> None:
        self._scenario_id = scenario_id
        self._rng = Random(self.seed)
        scenario = MOCK_SCENARIOS[scenario_id]

        supported_tools = [
            tool_name for tool_name in scenario.tool_effects if tool_name != "advance_time"
        ]
        self._rng.shuffle(supported_tools)

        max_actions = min(3, len(supported_tools))
        action_count = self._rng.randint(1, max_actions) if max_actions else 0
        selected_tools = supported_tools[:action_count]

        self._queue = [self._tool_action(tool_name) for tool_name in selected_tools]
        self._queue.append(action("advance_time", seconds=30))

    def select_action(self, observation: PulsePhysiologyObservation) -> ToolAction:
        if self._queue:
            return self._queue.pop(0)
        return self.fallback_action.model_copy(deep=True)

    def observe_outcome(self, action: ToolAction, result: EnvironmentResponse) -> None:
        return None

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
        prompt = build_policy_prompt(
            observation,
            available_tools=observation.available_tools,
            objective=self.objective,
            recent_history=self._recent_history,
        )
        allowed_tools = observation.available_tools or None
        try:
            raw_response = self.infer_fn(prompt)
            parsed_action = parse_tool_action(raw_response, allowed_tools=allowed_tools)
        except ToolParseError as exc:
            return self._fallback_action(observation, f"Model output parse failed: {exc}")
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            return self._fallback_action(observation, f"Model inference failed: {exc}")
        return self._apply_repeat_guard(parsed_action, observation)

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
        observation: PulsePhysiologyObservation,
        reason: str,
    ) -> ToolAction:
        available_tools = set(observation.available_tools or [])
        if observation.active_alerts and "check_deterioration" in available_tools:
            candidate = action("check_deterioration", reasoning=reason)
        elif "advance_time" in available_tools:
            candidate = action("advance_time", seconds=self.fallback_seconds, reasoning=reason)
        elif available_tools:
            fallback_tool = sorted(available_tools)[0]
            candidate = action(fallback_tool, reasoning=reason)
        else:
            candidate = action("advance_time", seconds=self.fallback_seconds, reasoning=reason)
        return self._apply_repeat_guard(candidate, observation)

    def _apply_repeat_guard(
        self,
        parsed_action: ToolAction,
        observation: PulsePhysiologyObservation,
    ) -> ToolAction:
        if self.anti_repeat_window <= 0 or len(self._recent_history) < self.anti_repeat_window:
            return parsed_action

        recent_tool_names = [
            item["action"]["tool_name"]
            for item in self._recent_history[-self.anti_repeat_window :]
        ]

        if all(tool_name == parsed_action.tool_name for tool_name in recent_tool_names):
            available_tools = set(observation.available_tools or [])
            if parsed_action.tool_name in READ_ONLY_TOOLS and "advance_time" in available_tools:
                return action(
                    "advance_time",
                    seconds=self.fallback_seconds,
                    reasoning=(
                        "Avoid repeating the same read-only tool without new information; "
                        "advance time to generate a fresh signal."
                    ),
                )
            if parsed_action.tool_name == "advance_time" and "check_deterioration" in available_tools:
                return action(
                    "check_deterioration",
                    reasoning=(
                        "Avoid repeating advance_time without reassessment; "
                        "check for new deterioration before acting again."
                    ),
                )

        return parsed_action


def build_expert_policy() -> ScriptedPolicy:
    """Factory for the deterministic expert baseline."""

    return ScriptedPolicy(name="expert", playbooks=EXPERT_PLAYBOOKS)


def build_no_action_policy() -> ScriptedPolicy:
    """Factory for the deterministic no-action baseline."""

    return ScriptedPolicy(name="no_action", playbooks=NO_ACTION_PLAYBOOKS)
