"""Demo the future prompt -> JSON tool call -> action loop on the mock backend.

Examples:

    python -m pulse_physiology_env.demo_llm_policy --scenario respiratory_distress
    python -m pulse_physiology_env.demo_llm_policy --scenario hemorrhagic_shock --max-steps 6
"""

from __future__ import annotations

import argparse
import json

from pulse_physiology_env.episode_runner import EpisodeRunner
from pulse_physiology_env.policies import LLMPolicy
from pulse_physiology_env.server.adapters import MockPulseAdapter
from pulse_physiology_env.server.mock_scenarios import DEFAULT_MOCK_SCENARIO_ID, MOCK_SCENARIOS
from pulse_physiology_env.trajectory_io import write_trace_json


def _extract_json_section(prompt: str, start_marker: str, end_marker: str) -> object:
    """Extract a JSON section from a rendered policy prompt."""

    start_idx = prompt.index(start_marker) + len(start_marker)
    end_idx = prompt.index(end_marker, start_idx)
    json_text = prompt[start_idx:end_idx].strip()
    return json.loads(json_text)


def _extract_snapshot(prompt: str) -> dict:
    """Extract the patient snapshot JSON from a rendered policy prompt."""

    return _extract_json_section(
        prompt,
        start_marker="Current patient snapshot:",
        end_marker="Recent decision history:",
    )


def _extract_recent_history(prompt: str) -> list[dict]:
    """Extract the recent decision history JSON from a rendered policy prompt."""

    return _extract_json_section(
        prompt,
        start_marker="Recent decision history:",
        end_marker="Available tools:",
    )


def _extract_available_tools(prompt: str) -> list[dict]:
    """Extract the tool catalog JSON from a rendered policy prompt."""

    return _extract_json_section(
        prompt,
        start_marker="Available tools:",
        end_marker="Rules:",
    )


def heuristic_infer_fn(prompt: str) -> str:
    """Simple stand-in for a future LLM tool-calling response."""

    snapshot = _extract_snapshot(prompt)
    recent_history = _extract_recent_history(prompt)
    alerts = set(snapshot.get("active_alerts") or [])
    scenario_id = snapshot.get("scenario_id")
    last_tool = recent_history[-1]["action"]["tool_name"] if recent_history else None
    second_last_tool = recent_history[-2]["action"]["tool_name"] if len(recent_history) > 1 else None
    recent_tools = [item["action"]["tool_name"] for item in recent_history[-2:]]
    available_tools = {tool["tool_name"] for tool in _extract_available_tools(prompt)}

    tool_name = "advance_time"
    arguments: dict = {"seconds": 30}
    reasoning = "Advance the scenario to observe the next physiological change."

    if (
        scenario_id == "hemorrhagic_shock"
        and "blood_loss" in alerts
        and "control_bleeding" in available_tools
        and "control_bleeding" not in recent_tools
    ):
        tool_name = "control_bleeding"
        arguments = {}
        reasoning = "Uncontrolled blood loss is an immediate driver of deterioration."
    elif (
        scenario_id == "hemorrhagic_shock"
        and "hypotension" in alerts
        and "give_fluids" in available_tools
        and last_tool != "give_fluids"
    ):
        tool_name = "give_fluids"
        arguments = {"volume_ml": 500}
        reasoning = "Low blood pressure suggests poor perfusion and fluid support may help."
    elif (
        scenario_id == "hemorrhagic_shock"
        and "tachycardia" in alerts
        and "give_fluids" in available_tools
        and "give_fluids" not in recent_tools
    ):
        tool_name = "give_fluids"
        arguments = {"volume_ml": 250}
        reasoning = "Persistent tachycardia after initial hemorrhage control suggests ongoing volume deficit."
    elif (
        scenario_id == "hemorrhagic_shock"
        and {"tachypnea", "hypoxemia"} & alerts
        and "give_oxygen" in available_tools
        and "give_oxygen" not in recent_tools
    ):
        tool_name = "give_oxygen"
        arguments = {"flow_lpm": 15}
        reasoning = "Oxygen support can reduce compensatory respiratory strain during hemorrhagic shock."
    elif (
        scenario_id == "hemorrhagic_shock"
        and {"tachypnea", "hypotension"} & alerts
        and "position_patient" in available_tools
        and "position_patient" not in recent_tools
    ):
        tool_name = "position_patient"
        arguments = {"position": "supine"}
        reasoning = "Supine positioning can support perfusion while hemorrhage is being corrected."
    elif "hypoxemia" in alerts and "give_oxygen" in available_tools and last_tool != "give_oxygen":
        tool_name = "give_oxygen"
        arguments = {"flow_lpm": 15}
        reasoning = "Low oxygen saturation should be treated with oxygen support first."
    elif "tachypnea" in alerts and "airway_support" in available_tools and last_tool != "airway_support":
        tool_name = "airway_support"
        arguments = {"mode": "basic"}
        reasoning = "Persistently high respiratory effort suggests the need for airway support."
    elif "tachypnea" in alerts and "position_patient" in available_tools and last_tool != "position_patient":
        tool_name = "position_patient"
        arguments = {"position": "upright"}
        reasoning = "Positioning may improve respiratory mechanics before further deterioration."
    elif (
        not alerts
        and last_tool in {"summarize_state", "check_deterioration"}
        and "advance_time" in available_tools
    ):
        tool_name = "advance_time"
        arguments = {"seconds": 30}
        reasoning = "The patient appears stable after reassessment, so advance time to confirm stability persists."
    elif not alerts and "summarize_state" in available_tools and last_tool != "summarize_state":
        tool_name = "summarize_state"
        arguments = {}
        reasoning = "The patient appears stable; summarize the state before moving forward."
    elif (
        "check_deterioration" in available_tools
        and last_tool != "check_deterioration"
        and second_last_tool != "check_deterioration"
    ):
        tool_name = "check_deterioration"
        arguments = {}
        reasoning = "Reassess whether the patient is actively worsening before advancing time."

    return json.dumps(
        {
            "tool_name": tool_name,
            "arguments": arguments,
            "reasoning": reasoning,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=DEFAULT_MOCK_SCENARIO_ID, choices=sorted(MOCK_SCENARIOS))
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--trace-json")
    args = parser.parse_args()

    backend = MockPulseAdapter(default_scenario_id=args.scenario)
    runner = EpisodeRunner(backend=backend, max_steps=args.max_steps)
    policy = LLMPolicy(infer_fn=heuristic_infer_fn, name="llm_demo")
    trace = runner.run(policy=policy, scenario_id=args.scenario)

    print("LLM-style policy demo")
    for key, value in trace.summary().items():
        print(f"  {key}: {value}")

    print("\nAction trace")
    for step in trace.steps:
        print(
            f"  step={step.step_index}"
            f" tool={step.action.tool_name}"
            f" reward={step.reward:.3f}"
            f" done={step.done}"
        )
        print(f"    reasoning={step.action.reasoning}")
        if step.tool_result is not None:
            print(f"    result={step.tool_result['message']}")
        if step.error is not None:
            print(f"    error={step.error['code']}: {step.error['message']}")

    if args.trace_json:
        write_trace_json(trace, args.trace_json)
        print(f"\nWrote JSON trace to {args.trace_json}")


if __name__ == "__main__":
    main()
