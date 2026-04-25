"""Typed patient-state contract for the Pulse-backed environment runtime."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LactateTrend = Literal["improving", "worsening", "stable"]


class ArterialBloodGasResult(BaseModel):
    """Clinically relevant arterial blood gas values."""

    model_config = ConfigDict(extra="forbid")

    ph: float | None = Field(default=None, description="Arterial blood pH.")
    partial_pressure_of_oxygen_mm_hg: float | None = Field(
        default=None,
        description="Arterial oxygen partial pressure.",
    )
    partial_pressure_of_carbon_dioxide_mm_hg: float | None = Field(
        default=None,
        description="Arterial carbon dioxide partial pressure.",
    )
    oxygen_saturation: float | None = Field(
        default=None,
        description="Arterial oxygen saturation as a 0-1 fraction.",
    )
    bicarbonate_meq_per_l: float | None = Field(
        default=None,
        description="Derived bicarbonate concentration.",
    )
    lactate_mg_per_dl: float | None = Field(
        default=None,
        description="Arterial lactate concentration.",
    )
    base_excess_meq_per_l: float | None = Field(
        default=None,
        description="Arterial base excess.",
    )
    base_deficit_meq_per_l: float | None = Field(
        default=None,
        description="Positive base deficit derived from base excess.",
    )


class CompleteBloodCountResult(BaseModel):
    """Subset of CBC values used by the hackathon spec."""

    model_config = ConfigDict(extra="forbid")

    hemoglobin_g_per_dl: float | None = Field(
        default=None,
        description="Estimated hemoglobin concentration.",
    )
    hematocrit_fraction: float | None = Field(
        default=None,
        description="Hematocrit as a 0-1 fraction.",
    )
    white_blood_cell_count_per_u_l: float | None = Field(
        default=None,
        description="White blood cell count.",
    )
    platelet_count_per_u_l: float | None = Field(
        default=None,
        description="Platelet count when available from Pulse.",
    )
    red_blood_cell_count_per_u_l: float | None = Field(
        default=None,
        description="Red blood cell count.",
    )


class BasicMetabolicPanelResult(BaseModel):
    """Core metabolic panel values needed for the early toolset."""

    model_config = ConfigDict(extra="forbid")

    sodium_mmol_per_l: float | None = Field(default=None)
    potassium_mmol_per_l: float | None = Field(default=None)
    calcium_mmol_per_l: float | None = Field(default=None)
    creatinine_mg_per_dl: float | None = Field(default=None)
    glucose_mg_per_dl: float | None = Field(default=None)


class PatientState(BaseModel):
    """Stable runtime view of the simulated patient."""

    model_config = ConfigDict(extra="forbid")

    heart_rate_bpm: float | None = Field(default=None)
    systolic_bp_mm_hg: float | None = Field(default=None)
    diastolic_bp_mm_hg: float | None = Field(default=None)
    mean_arterial_pressure_mm_hg: float | None = Field(default=None)
    cardiac_output_l_per_min: float | None = Field(default=None)

    spo2: float | None = Field(default=None)
    respiratory_rate_bpm: float | None = Field(default=None)
    etco2_mm_hg: float | None = Field(default=None)
    tidal_volume_ml: float | None = Field(default=None)
    breath_sounds: str = Field(
        default="present bilateral",
        description="Derived bedside breath-sound summary.",
    )

    consciousness_level: float = Field(
        default=1.0,
        description="0-1 scale where 1 is fully conscious.",
    )
    pain_level: float = Field(
        default=0.0,
        description="0-1 heuristic pain scale.",
    )

    core_temperature_c: float | None = Field(default=None)

    abg_result: ArterialBloodGasResult = Field(default_factory=ArterialBloodGasResult)
    cbc_result: CompleteBloodCountResult = Field(default_factory=CompleteBloodCountResult)
    bmp_result: BasicMetabolicPanelResult = Field(default_factory=BasicMetabolicPanelResult)

    pending_diagnostics: dict[str, int] = Field(
        default_factory=dict,
        description="Diagnostic tool name to remaining timesteps.",
    )
    active_infusions: dict[str, float] = Field(
        default_factory=dict,
        description="Drug name to current delivered dose estimate.",
    )
    active_hemorrhages: dict[str, float] = Field(
        default_factory=dict,
        description="Active hemorrhage site to flow rate in mL/min.",
    )

    shock_index: float | None = Field(default=None)
    lactate_trend: LactateTrend = Field(default="stable")
    intubated: bool = Field(default=False)

    simulated_time_seconds: float = Field(default=0.0)
    action_budget_remaining: int | None = Field(default=None)
