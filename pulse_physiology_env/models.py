"""Action, observation, and tool-result models for the Pulse environment."""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, ConfigDict, Field

from .patient_state import (
    ArterialBloodGasResult,
    BasicMetabolicPanelResult,
    CompleteBloodCountResult,
    LactateTrend,
    MentalStatus,
    PatientState,
)


class PulsePhysiologyAction(Action):
    """Tool invocation sent from the agent or client."""

    tool_name: str = Field(..., description="Name of the tool to execute.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured arguments for the named tool.",
    )
    reasoning: str | None = Field(
        default=None,
        description="Optional human-readable rationale for the action.",
    )


class ToolResult(BaseModel):
    """Structured result returned for every handled tool call."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    success: bool
    message: str
    state_changed: bool
    changed_fields: list[str] = Field(default_factory=list)


class ToolError(BaseModel):
    """Structured error attached to failed tool calls."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    retryable: bool


class PulsePhysiologyObservation(Observation):
    """Observation returned by the Pulse physiology environment."""

    scenario_id: str = Field(default="baseline")
    patient_id: str = Field(default="standard_male")
    sim_time_s: float = Field(default=0.0)

    heart_rate_bpm: float | None = Field(default=None)
    systolic_bp_mmhg: float | None = Field(default=None)
    diastolic_bp_mmhg: float | None = Field(default=None)
    mean_arterial_pressure_mmhg: float | None = Field(default=None)
    cardiac_output_l_per_min: float | None = Field(default=None)

    spo2: float | None = Field(default=None)
    respiration_rate_bpm: float | None = Field(default=None)
    blood_volume_ml: float | None = Field(default=None)
    mental_status: MentalStatus = Field(default="alert")
    active_alerts: list[str] = Field(default_factory=list)

    etco2_mmhg: float | None = Field(default=None)
    tidal_volume_ml: float | None = Field(default=None)
    breath_sounds: str = Field(default="present bilateral")
    core_temperature_c: float | None = Field(default=None)

    shock_index: float | None = Field(default=None)
    lactate_trend: LactateTrend = Field(default="stable")

    position: str = Field(default="supine")
    oxygen_device: str | None = Field(default=None)
    oxygen_flow_lpm: float | None = Field(default=None)
    airway_support: str | None = Field(default=None)
    intubated: bool = Field(default=False)

    abg_result: ArterialBloodGasResult = Field(default_factory=ArterialBloodGasResult)
    cbc_result: CompleteBloodCountResult = Field(default_factory=CompleteBloodCountResult)
    bmp_result: BasicMetabolicPanelResult = Field(default_factory=BasicMetabolicPanelResult)

    pending_diagnostics: dict[str, int] = Field(default_factory=dict)
    active_infusions: dict[str, float] = Field(default_factory=dict)
    active_hemorrhages: dict[str, float] = Field(default_factory=dict)

    available_tools: list[str] = Field(default_factory=list)
    tool_result: ToolResult | None = Field(default=None)
    error: ToolError | None = Field(default=None)

    @classmethod
    def from_patient_state(
        cls,
        state: PatientState,
        *,
        reward: float | None,
        available_tools: list[str],
        tool_result: ToolResult | None,
        error: ToolError | None,
        metadata: dict[str, Any] | None = None,
    ) -> "PulsePhysiologyObservation":
        payload = state.model_dump()
        payload.update(
            reward=reward,
            done=state.done,
            available_tools=available_tools,
            tool_result=tool_result,
            error=error,
            metadata=metadata or {},
        )
        return cls(**payload)
