"""Evaluation harness for comparing mock policies.

Run from the repo root with:

    python -m pulse_physiology_env.eval_mock
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from pulse_physiology_env.demo_llm_policy import heuristic_infer_fn
from pulse_physiology_env.episode_runner import EpisodeRunner, EpisodeTerminationReason
from pulse_physiology_env.models import EnvironmentResponse, ObservationMetadata, PulsePhysiologyObservation, ToolError, ToolResult
from pulse_physiology_env.policies import (
    LLMPolicy,
    RandomPolicy,
    action,
    build_expert_policy,
    build_no_action_policy,
)
from pulse_physiology_env.prompt_builder import build_policy_prompt
from pulse_physiology_env.tool_availability import ToolAvailabilityError
from pulse_physiology_env.tool_parser import ParseError, ParseWarning, parse_tool_action, parse_with_fallback
from pulse_physiology_env.server.adapters import MockPulseAdapter
from pulse_physiology_env.server.mock_scenarios import MOCK_SCENARIOS
from pulse_physiology_env.patient_state import PatientState


@dataclass(frozen=True)
class PolicyScore:
    """Summary score for one policy over all scenarios."""

    policy_name: str
    per_scenario: dict[str, float]

    @property
    def average_reward(self) -> float:
        return round(sum(self.per_scenario.values()) / len(self.per_scenario), 3)


def _make_observation(*, available_tools: list[str]) -> PulsePhysiologyObservation:
    """Create a compact observation fixture for consumer-side regression checks."""

    return PulsePhysiologyObservation(
        scenario_id="baseline_stable",
        patient_id="regression_patient",
        sim_time_s=0.0,
        heart_rate_bpm=72.0,
        systolic_bp_mmhg=118.0,
        diastolic_bp_mmhg=76.0,
        spo2=0.98,
        respiration_rate_bpm=14.0,
        blood_volume_ml=5500.0,
        available_tools=available_tools,
    )


def _make_response(
    observation: PulsePhysiologyObservation,
    *,
    reward: float,
    error: ToolError | None = None,
    tool_result: ToolResult | None = None,
) -> EnvironmentResponse:
    """Wrap a fixture observation in the standard environment response envelope."""

    return EnvironmentResponse(
        observation=observation,
        reward=reward,
        done=observation.done,
        metadata=ObservationMetadata(step_count=0, available_tools=list(observation.available_tools)),
        tool_result=tool_result,
        error=error,
    )


class _FixedActionPolicy:
    """Deterministic policy used to probe episode-runner edge cases."""

    name = "fixed_action"

    def __init__(self, tool_name: str = "get_vitals") -> None:
        self._action = action(tool_name)

    def reset(self, scenario_id: str) -> None:
        return None

    def select_action(self, observation: PulsePhysiologyObservation):
        return self._action

    def observe_outcome(self, action, result) -> None:
        return None


class _TransientFailureBackend:
    """Backend stub that succeeds only after transient retryable failures."""

    def __init__(self) -> None:
        self.attempts = 0
        self._state = PatientState(
            scenario_id="baseline_stable",
            patient_id="retry_backend",
            heart_rate_bpm=72.0,
            systolic_bp_mmhg=118.0,
            diastolic_bp_mmhg=76.0,
            spo2=0.98,
            respiration_rate_bpm=14.0,
            blood_volume_ml=5500.0,
        )

    def reset(self, scenario_id: str | None = None) -> EnvironmentResponse:
        return _make_response(_make_observation(available_tools=["get_vitals", "advance_time"]), reward=0.0)

    def step(self, tool_action) -> EnvironmentResponse:
        self.attempts += 1
        observation = _make_observation(available_tools=["get_vitals", "advance_time"])
        if self.attempts < 3:
            return _make_response(
                observation,
                reward=-1.0,
                error=ToolError(
                    code="TEMPORARY_BUSY",
                    message="Transient backend failure.",
                    retryable=True,
                ),
                tool_result=ToolResult(
                    tool_name=tool_action.tool_name,
                    success=False,
                    message="Transient backend failure.",
                    state_changed=False,
                    changed_fields=[],
                ),
            )

        return _make_response(
            observation,
            reward=0.25,
            tool_result=ToolResult(
                tool_name=tool_action.tool_name,
                success=True,
                message="Recovered after retries.",
                state_changed=False,
                changed_fields=[],
            ),
        )

    def get_state(self) -> PatientState:
        return self._state


class _MissingToolsBackend:
    """Backend stub that violates the available-tools contract on reset."""

    def __init__(self) -> None:
        self._state = PatientState(
            scenario_id="baseline_stable",
            patient_id="missing_tools_backend",
            heart_rate_bpm=72.0,
            systolic_bp_mmhg=118.0,
            diastolic_bp_mmhg=76.0,
            spo2=0.98,
            respiration_rate_bpm=14.0,
            blood_volume_ml=5500.0,
        )

    def reset(self, scenario_id: str | None = None) -> EnvironmentResponse:
        return _make_response(_make_observation(available_tools=[]), reward=0.0)

    def step(self, tool_action) -> EnvironmentResponse:
        raise RuntimeError("step() should not be called when available_tools is missing.")

    def get_state(self) -> PatientState:
        return self._state


def _regression_check_retryable_runner_behavior() -> None:
    """Ensure retryable backend failures do not terminate the episode immediately."""

    runner = EpisodeRunner(
        backend=_TransientFailureBackend(),
        max_steps=1,
        max_retry_attempts=3,
        retry_backoff_s=0.0,
    )
    trace = runner.run(policy=_FixedActionPolicy(), scenario_id="baseline_stable")

    if trace.termination_reason != EpisodeTerminationReason.MAX_TIMESTEPS:
        raise SystemExit(
            "Episode runner regression: retryable failures should recover without fatal termination."
        )
    if trace.num_steps != 1 or trace.steps[0].error is not None:
        raise SystemExit(
            "Episode runner regression: the recovered step should be recorded as a successful action."
        )
    if "Retryable error on get_vitals attempt 1/3" not in "\n".join(trace.events):
        raise SystemExit("Episode runner regression: retryable attempts were not logged.")


def _regression_check_available_tools_fail_closed() -> None:
    """Ensure missing available tools stop policy execution instead of exposing the full catalog."""

    observation = _make_observation(available_tools=[])
    try:
        build_policy_prompt(observation)
    except ToolAvailabilityError:
        pass
    else:  # pragma: no cover - exercised by CLI validation
        raise SystemExit("Prompt builder regression: missing available_tools should fail closed.")

    runner = EpisodeRunner(backend=_MissingToolsBackend(), max_steps=1, retry_backoff_s=0.0)
    trace = runner.run(policy=_FixedActionPolicy(), scenario_id="baseline_stable")
    if trace.termination_reason != EpisodeTerminationReason.FATAL_BACKEND_ERROR:
        raise SystemExit(
            "Episode runner regression: missing available_tools must terminate as a fatal backend error."
        )
    if trace.num_steps != 0:
        raise SystemExit("Episode runner regression: missing available_tools should terminate before any step runs.")


def _regression_check_tool_parser_hierarchy() -> None:
    """Ensure parser extraction prefers schema-aware JSON candidates over greedy braces."""

    fenced_payload = parse_with_fallback(
        'Narration first\n```json\n{"tool_name":"get_vitals","arguments":{},"reasoning":"check now"}\n```'
    )
    if fenced_payload["tool_name"] != "get_vitals":
        raise SystemExit("Tool parser regression: fenced JSON extraction failed.")

    action_payload = parse_tool_action(
        'Model says: {"tool_name":"give_oxygen","arguments":{"flow_lpm":15},"reasoning":"low saturation"} '
        'extra trailing text {"noise":true}'
    )
    if action_payload.tool_name != "give_oxygen" or action_payload.arguments.get("flow_lpm") != 15.0:
        raise SystemExit("Tool parser regression: schema-aware extraction failed before greedy fallback.")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ParseWarning)
        try:
            parse_with_fallback('Prefix only {"note": 1}', log_warnings=True)
        except ParseError as exc:
            if "Raw output:" not in str(exc):
                raise SystemExit("Tool parser regression: parse errors must include the raw model output preview.")
        else:  # pragma: no cover - exercised by CLI validation
            raise SystemExit("Tool parser regression: invalid fallback payload should raise ParseError.")

    if not any(isinstance(warning.message, ParseWarning) for warning in caught):
        raise SystemExit("Tool parser regression: fallback parsing must emit ParseWarning.")


def score_policy(policy_factory, policy_name: str) -> PolicyScore:
    """Evaluate one policy factory across all mock scenarios."""

    per_scenario: dict[str, float] = {}
    for scenario_id in MOCK_SCENARIOS:
        backend = MockPulseAdapter(default_scenario_id=scenario_id)
        runner = EpisodeRunner(backend=backend, max_steps=8)
        policy = policy_factory(scenario_id)
        trace = runner.run(policy=policy, scenario_id=scenario_id)
        per_scenario[scenario_id] = trace.total_reward

    return PolicyScore(policy_name=policy_name, per_scenario=per_scenario)


def score_random_policy(num_seeds: int = 12) -> PolicyScore:
    """Evaluate the mean reward of seeded random policies."""

    per_scenario: dict[str, float] = {}
    for scenario_id in MOCK_SCENARIOS:
        rewards = []
        for seed in range(num_seeds):
            backend = MockPulseAdapter(default_scenario_id=scenario_id)
            runner = EpisodeRunner(backend=backend, max_steps=8)
            policy = RandomPolicy(seed=seed)
            trace = runner.run(policy=policy, scenario_id=scenario_id)
            rewards.append(trace.total_reward)
        per_scenario[scenario_id] = round(sum(rewards) / len(rewards), 3)

    return PolicyScore(policy_name="random", per_scenario=per_scenario)


def print_policy_score(score: PolicyScore) -> None:
    """Pretty-print one policy summary."""

    print(f"{score.policy_name} policy")
    for scenario_id, reward in score.per_scenario.items():
        print(f"  {scenario_id}: {reward:.3f}")
    print(f"  average: {score.average_reward:.3f}")


def main() -> None:
    """Compare expert, random, and no-action baselines."""

    print("Consumer regression checks\n")
    _regression_check_retryable_runner_behavior()
    print("PASS retryable runner handling")
    _regression_check_available_tools_fail_closed()
    print("PASS available_tools fail-closed handling\n")
    _regression_check_tool_parser_hierarchy()
    print("PASS tool parser hierarchy\n")

    expert = score_policy(lambda scenario_id: build_expert_policy(), "expert")
    llm_demo = score_policy(
        lambda scenario_id: LLMPolicy(infer_fn=heuristic_infer_fn, name="llm_demo"),
        "llm_demo",
    )
    random_policy = score_random_policy()
    no_action = score_policy(lambda scenario_id: build_no_action_policy(), "no_action")

    print("Mock policy evaluation\n")
    print_policy_score(expert)
    print()
    print_policy_score(llm_demo)
    print()
    print_policy_score(random_policy)
    print()
    print_policy_score(no_action)
    print()

    if not (
        expert.average_reward > llm_demo.average_reward > random_policy.average_reward > no_action.average_reward
    ):
        raise SystemExit(
            "Policy ranking check failed: expected expert > llm_demo > random > no_action on average."
        )

    print("PASS expert > llm_demo > random > no_action")


if __name__ == "__main__":
    main()
