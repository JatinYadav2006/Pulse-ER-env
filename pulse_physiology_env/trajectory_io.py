"""Trajectory export utilities for Pulse-ER episode traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .episode_runner import EpisodeTrace


def _observation_to_dict(observation) -> dict[str, Any]:
    payload = observation.model_dump()
    mental_status = payload.get("mental_status")
    if hasattr(mental_status, "value"):
        payload["mental_status"] = mental_status.value
    return payload


def episode_trace_to_dict(trace: EpisodeTrace) -> dict[str, Any]:
    """Convert an episode trace into a JSON-serializable dictionary."""

    return {
        "scenario_id": trace.scenario_id,
        "policy_name": trace.policy_name,
        "total_reward": trace.total_reward,
        "num_steps": trace.num_steps,
        "termination_reason": trace.termination_reason.value,
        "action_budget_remaining": trace.action_budget_remaining,
        "events": list(trace.events),
        "initial_observation": _observation_to_dict(trace.initial_observation),
        "final_observation": _observation_to_dict(trace.final_observation),
        "steps": [
            {
                "step_index": step.step_index,
                "action": step.action.model_dump(),
                "reward": step.reward,
                "done": step.done,
                "observation": _observation_to_dict(step.observation),
                "tool_result": step.tool_result,
                "error": step.error,
            }
            for step in trace.steps
        ],
    }


def write_trace_json(trace: EpisodeTrace, path: str | Path) -> Path:
    """Write one episode trace to a JSON file."""

    output_path = Path(path)
    output_path.write_text(json.dumps(episode_trace_to_dict(trace), indent=2), encoding="utf-8")
    return output_path


def append_trace_jsonl(trace: EpisodeTrace, path: str | Path) -> Path:
    """Append one episode trace to a JSONL file."""

    output_path = Path(path)
    line = json.dumps(episode_trace_to_dict(trace), separators=(",", ":"))
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return output_path
