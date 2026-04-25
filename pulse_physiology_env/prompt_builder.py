"""Prompt construction utilities for future LLM-driven policies."""

from __future__ import annotations

import json
from typing import Any

from .models import PulsePhysiologyObservation
from .tool_catalog import INITIAL_TOOL_NAMES, build_tool_catalog as build_shared_tool_catalog


def _mental_status_value(mental_status: Any) -> str:
    return getattr(mental_status, "value", str(mental_status))


def observation_snapshot(observation: PulsePhysiologyObservation) -> dict[str, Any]:
    """Extract a concise but useful observation snapshot for prompting."""

    return {
        "scenario_id": observation.scenario_id,
        "patient_id": observation.patient_id,
        "sim_time_s": observation.sim_time_s,
        "heart_rate_bpm": observation.heart_rate_bpm,
        "systolic_bp_mmhg": observation.systolic_bp_mmhg,
        "diastolic_bp_mmhg": observation.diastolic_bp_mmhg,
        "spo2": observation.spo2,
        "respiration_rate_bpm": observation.respiration_rate_bpm,
        "blood_volume_ml": observation.blood_volume_ml,
        "mental_status": _mental_status_value(observation.mental_status),
        "active_alerts": observation.active_alerts,
        "done": observation.done,
        "position": observation.position,
        "oxygen_device": observation.oxygen_device,
        "oxygen_flow_lpm": observation.oxygen_flow_lpm,
        "airway_support": observation.airway_support,
        "tool_result": observation.tool_result.model_dump() if observation.tool_result else None,
        "error": observation.error.model_dump() if observation.error else None,
    }


def build_tool_catalog(available_tools: list[str] | None = None) -> list[dict[str, Any]]:
    """Build a prompt-safe catalog from the shared tool registry."""

    return build_shared_tool_catalog(available_tools or list(INITIAL_TOOL_NAMES))


def build_policy_prompt(
    observation: PulsePhysiologyObservation,
    *,
    available_tools: list[str] | None = None,
    objective: str | None = None,
    recent_history: list[dict[str, Any]] | None = None,
) -> str:
    """Render a prompt asking a model to choose the next tool action."""

    snapshot = observation_snapshot(observation)
    tool_catalog = build_tool_catalog(available_tools or observation.available_tools)
    history = recent_history or []
    objective_text = objective or (
        "Choose the single best next tool call to stabilize the patient and avoid deterioration."
    )

    return "\n".join(
        [
            "You are a clinical reasoning agent operating in Pulse-ER.",
            objective_text,
            "",
            "Return exactly one JSON object with this schema:",
            '{',
            '  "tool_name": "<one supported tool name>",',
            '  "arguments": { ... },',
            '  "reasoning": "<short justification>"',
            '}',
            "",
            "Current patient snapshot:",
            json.dumps(snapshot, indent=2),
            "",
            "Recent decision history:",
            json.dumps(history, indent=2),
            "",
            "Available tools:",
            json.dumps(tool_catalog, indent=2),
            "",
            "Rules:",
            "- Output JSON only.",
            "- Use only supported tool names.",
            "- Keep arguments structured, not prose.",
            "- Include every required argument for the selected tool.",
            "- Prefer actions that address active alerts or worsening vitals.",
            "- Avoid repeating the same read-only tool when nothing material has changed.",
        ]
    )
