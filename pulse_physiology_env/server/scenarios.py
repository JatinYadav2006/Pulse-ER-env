"""Scenario presets used to initialize Pulse-backed hackathon cases."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from .pulse_engine_adapter import PulseEngineAdapter


@dataclass(frozen=True)
class ScenarioDefinition:
    """High-level scenario configuration for environment reset."""

    scenario_id: str
    description: str
    patient_id: str = "standard_male"
    state_file: str = "StandardMale@0s.json"
    max_time_s: float = 1800.0
    setup: Callable[[PulseEngineAdapter], None] | None = None


def _setup_respiratory_distress(adapter: PulseEngineAdapter) -> None:
    adapter.set_tension_pneumothorax("left", severity=0.75, advance_time_seconds=90.0)


def _setup_hemorrhagic_shock(adapter: PulseEngineAdapter) -> None:
    adapter.set_hemorrhage("right_leg", flow_rate_ml_per_min=180.0)
    adapter.advance_time(300.0)


def _setup_polytrauma_demo(adapter: PulseEngineAdapter) -> None:
    adapter.set_tension_pneumothorax("left", severity=0.8)
    adapter.set_hemorrhage("right_leg", flow_rate_ml_per_min=180.0)
    adapter.advance_time(180.0)


SCENARIOS: dict[str, ScenarioDefinition] = {
    "respiratory_distress": ScenarioDefinition(
        scenario_id="respiratory_distress",
        description="Hypoxemic respiratory distress from a left tension pneumothorax.",
        setup=_setup_respiratory_distress,
    ),
    "hemorrhagic_shock": ScenarioDefinition(
        scenario_id="hemorrhagic_shock",
        description="Progressive hemorrhagic shock from uncontrolled right leg bleeding.",
        setup=_setup_hemorrhagic_shock,
    ),
    "polytrauma_demo": ScenarioDefinition(
        scenario_id="polytrauma_demo",
        description="Combined hemorrhagic shock and tension pneumothorax for the hackathon demo.",
        max_time_s=2400.0,
        setup=_setup_polytrauma_demo,
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
