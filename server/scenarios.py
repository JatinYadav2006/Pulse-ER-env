"""Scenario presets used to initialize Pulse-backed hackathon cases."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable

from pulse_physiology_env.patient_state import ScenarioDifficulty

from .pulse_engine_adapter import PulseEngineAdapter


def _patient_id_from_state_file(state_file: str) -> str:
    stem = state_file.split("@", 1)[0]
    first_pass = []
    for index, char in enumerate(stem):
        if index > 0 and char.isupper() and (not stem[index - 1].isupper()):
            first_pass.append("_")
        first_pass.append(char.lower())
    return "".join(first_pass)


@dataclass(frozen=True)
class PatientProfile:
    """A patient baseline that can seed a scenario reset."""

    state_file: str
    patient_id: str


@dataclass(frozen=True)
class ScenarioDefinition:
    """High-level scenario configuration for environment reset."""

    scenario_id: str
    description: str
    difficulty: ScenarioDifficulty
    reward_profile: str
    patient_pool: tuple[PatientProfile, ...]
    max_time_s: float = 1800.0
    setup: Callable[[PulseEngineAdapter], None] | None = None

    def choose_patient(self, rng: random.Random) -> PatientProfile:
        return rng.choice(self.patient_pool)


def _profiles(*state_files: str) -> tuple[PatientProfile, ...]:
    return tuple(
        PatientProfile(
            state_file=state_file,
            patient_id=_patient_id_from_state_file(state_file),
        )
        for state_file in state_files
    )


EASY_PATIENTS = _profiles(
    "Bradycardic@0s.json",
    "Nathan@0s.json",
    "StandardMale@0s.json",
    "DefaultMale@0s.json",
    "Overweight@0s.json",
    "Carol@0s.json",
    "Jeff@0s.json",
)

MEDIUM_PATIENTS = _profiles(
    "Jane@0s.json",
    "Cynthia@0s.json",
    "Underweight@0s.json",
    "DefaultFemale@0s.json",
    "Rick@0s.json",
    "Soldier@0s.json",
    "ExtremeMale@0s.json",
)

HARD_PATIENTS = _profiles(
    "StandardFemale@0s.json",
    "Joel@0s.json",
    "Tachycardic@0s.json",
    "ExtremeFemale@0s.json",
    "Gus@0s.json",
    "Hassan@0s.json",
)


def _setup_easy_soldier(adapter: PulseEngineAdapter) -> None:
    adapter.set_tension_pneumothorax("left", severity=0.45)
    adapter.set_hemorrhage("right_leg", flow_rate_ml_per_min=110.0)
    adapter.advance_time(90.0)


def _setup_medium_carol(adapter: PulseEngineAdapter) -> None:
    adapter.set_tension_pneumothorax("left", severity=0.65)
    adapter.set_hemorrhage("right_leg", flow_rate_ml_per_min=140.0)
    adapter.set_hemorrhage(
        "spleen",
        flow_rate_ml_per_min=45.0,
        hemorrhage_type="internal",
    )
    adapter.advance_time(150.0)


def _setup_hard_underweight(adapter: PulseEngineAdapter) -> None:
    adapter.set_tension_pneumothorax("left", severity=0.8)
    adapter.set_hemorrhage("right_leg", flow_rate_ml_per_min=180.0)
    adapter.set_hemorrhage(
        "spleen",
        flow_rate_ml_per_min=80.0,
        hemorrhage_type="internal",
    )
    adapter.advance_time(210.0)


SCENARIOS: dict[str, ScenarioDefinition] = {
    "trauma_easy_soldier": ScenarioDefinition(
        scenario_id="trauma_easy_soldier",
        description=(
            "Easy trauma bucket. One of seven baselines with the strongest measured resilience "
            "under a standardized trauma challenge is chosen at reset, then loaded into a left "
            "tension pneumothorax plus controllable right leg hemorrhage case."
        ),
        difficulty="easy",
        reward_profile="polytrauma",
        patient_pool=EASY_PATIENTS,
        max_time_s=1800.0,
        setup=_setup_easy_soldier,
    ),
    "trauma_medium_carol": ScenarioDefinition(
        scenario_id="trauma_medium_carol",
        description=(
            "Medium trauma bucket. One of seven baselines from the middle resilience band is "
            "chosen at reset, then loaded into chest trauma with mixed external and occult "
            "internal bleeding."
        ),
        difficulty="medium",
        reward_profile="polytrauma",
        patient_pool=MEDIUM_PATIENTS,
        max_time_s=2100.0,
        setup=_setup_medium_carol,
    ),
    "trauma_hard_underweight": ScenarioDefinition(
        scenario_id="trauma_hard_underweight",
        description=(
            "Hard trauma bucket. One of six baselines with the lowest measured resilience under "
            "the same standardized trauma challenge is chosen at reset, then loaded into severe "
            "fast-decompensating polytrauma."
        ),
        difficulty="hard",
        reward_profile="polytrauma",
        patient_pool=HARD_PATIENTS,
        max_time_s=1800.0,
        setup=_setup_hard_underweight,
    ),
    "polytrauma_demo": ScenarioDefinition(
        scenario_id="polytrauma_demo",
        description=(
            "Hackathon demo alias for the medium trauma bucket with randomized medium-difficulty baselines."
        ),
        difficulty="medium",
        reward_profile="polytrauma",
        patient_pool=MEDIUM_PATIENTS,
        max_time_s=2100.0,
        setup=_setup_medium_carol,
    ),
}


DEFAULT_SCENARIO_ID = os.getenv("PULSE_DEFAULT_SCENARIO", "polytrauma_demo")


def get_scenario_definition(scenario_id: str | None) -> ScenarioDefinition:
    """Resolve a scenario identifier to a configured scenario definition."""

    requested = (scenario_id or DEFAULT_SCENARIO_ID).strip()
    if requested not in SCENARIOS:
        valid = ", ".join(sorted(SCENARIOS))
        raise KeyError(f"Unknown scenario_id '{requested}'. Expected one of: {valid}")
    return SCENARIOS[requested]
