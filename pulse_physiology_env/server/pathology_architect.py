"""Minimal scenario authoring for generated trauma cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from pulse_physiology_env.patient_state import ScenarioDifficulty

from .pulse_engine_adapter import PulseEngineAdapter
from .scenarios import (
    EASY_PATIENTS,
    HARD_PATIENTS,
    MEDIUM_PATIENTS,
    PatientProfile,
    ScenarioDefinition,
)

InjuryType = Literal["tension_pneumothorax", "hemorrhagic_shock", "cardiac_tamponade", "polytrauma"]


@dataclass(frozen=True)
class PathologyBlueprint:
    """Generated case definition that can be applied directly to Pulse."""

    scenario_id: str
    description: str
    patient_id: str
    state_file: str
    injury_type: InjuryType
    severity: float
    difficulty: ScenarioDifficulty
    reward_profile: str
    max_time_s: float
    setup_actions: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable blueprint."""

        return asdict(self)


class PathologyArchitect:
    """Creates generated scenario blueprints from a small authoring surface."""

    _PATIENTS = {
        profile.patient_id: profile
        for profile in (*EASY_PATIENTS, *MEDIUM_PATIENTS, *HARD_PATIENTS)
    }

    def supported_patients(self) -> list[str]:
        """Return patient ids that can be used for generated cases."""

        return sorted(self._PATIENTS)

    @staticmethod
    def supported_injury_types() -> list[str]:
        """Return supported generated injury families."""

        return ["cardiac_tamponade", "hemorrhagic_shock", "polytrauma", "tension_pneumothorax"]

    def build_blueprint(
        self,
        *,
        patient_id: str,
        injury_type: str,
        severity: float,
    ) -> PathologyBlueprint:
        """Build a generated case from a single injury type and severity value."""

        profile = self._resolve_patient(patient_id)
        injury_key = injury_type.strip().lower().replace("-", "_").replace(" ", "_")
        if injury_key not in self.supported_injury_types():
            valid = ", ".join(self.supported_injury_types())
            raise ValueError(f"Unsupported injury_type '{injury_type}'. Expected one of: {valid}")

        clamped_severity = max(0.0, min(1.0, float(severity)))
        difficulty = self._difficulty_from_severity(clamped_severity)
        reward_profile = "polytrauma" if injury_key == "polytrauma" else injury_key
        setup_actions = self._build_setup_actions(injury_key, clamped_severity)
        scenario_id = f"generated_{injury_key}_{profile.patient_id}"
        description = (
            f"Generated {injury_key.replace('_', ' ')} case for {profile.patient_id} at severity "
            f"{clamped_severity:.2f}."
        )
        max_time_s = 1800.0 if difficulty != "hard" else 2100.0
        return PathologyBlueprint(
            scenario_id=scenario_id,
            description=description,
            patient_id=profile.patient_id,
            state_file=profile.state_file,
            injury_type=injury_key,  # type: ignore[arg-type]
            severity=clamped_severity,
            difficulty=difficulty,
            reward_profile=reward_profile,
            max_time_s=max_time_s,
            setup_actions=tuple(setup_actions),
        )

    def apply_blueprint(self, adapter: PulseEngineAdapter, blueprint: PathologyBlueprint) -> None:
        """Apply a generated blueprint to the current Pulse patient."""

        for step in blueprint.setup_actions:
            action_name = step["action"]
            if action_name == "set_tension_pneumothorax":
                adapter.set_tension_pneumothorax(
                    step["side"],
                    severity=float(step["severity"]),
                )
            elif action_name == "set_hemorrhage":
                adapter.set_hemorrhage(
                    step["compartment"],
                    flow_rate_ml_per_min=float(step["flow_rate_ml_per_min"]),
                    hemorrhage_type=str(step.get("hemorrhage_type", "external")),
                )
            elif action_name == "set_pericardial_effusion":
                adapter.set_pericardial_effusion(
                    effusion_rate_ml_per_min=float(step["effusion_rate_ml_per_min"]),
                )
            elif action_name == "advance_time":
                adapter.advance_time(float(step["seconds"]))
            else:
                raise ValueError(f"Unsupported setup action '{action_name}' in generated blueprint.")

    def to_scenario_definition(self, blueprint: PathologyBlueprint) -> ScenarioDefinition:
        """Convert a generated blueprint into a resettable ScenarioDefinition."""

        patient = self._resolve_patient(blueprint.patient_id)
        return ScenarioDefinition(
            scenario_id=blueprint.scenario_id,
            description=blueprint.description,
            difficulty=blueprint.difficulty,
            reward_profile=blueprint.reward_profile,
            patient_pool=(patient,),
            max_time_s=blueprint.max_time_s,
            setup=lambda adapter, blueprint=blueprint: self.apply_blueprint(adapter, blueprint),
        )

    def _resolve_patient(self, patient_id: str) -> PatientProfile:
        patient_key = patient_id.strip().lower().replace("-", "_").replace(" ", "_")
        if patient_key not in self._PATIENTS:
            valid = ", ".join(sorted(self._PATIENTS))
            raise ValueError(f"Unknown patient_id '{patient_id}'. Expected one of: {valid}")
        return self._PATIENTS[patient_key]

    @staticmethod
    def _difficulty_from_severity(severity: float) -> ScenarioDifficulty:
        if severity < 0.34:
            return "easy"
        if severity < 0.67:
            return "medium"
        return "hard"

    def _build_setup_actions(self, injury_type: str, severity: float) -> list[dict[str, Any]]:
        if injury_type == "tension_pneumothorax":
            return [
                {"action": "set_tension_pneumothorax", "side": "left", "severity": round(0.30 + 0.55 * severity, 3)},
                {"action": "advance_time", "seconds": round(45.0 + 135.0 * severity, 1)},
            ]
        if injury_type == "hemorrhagic_shock":
            steps: list[dict[str, Any]] = [
                {
                    "action": "set_hemorrhage",
                    "compartment": "right_leg",
                    "hemorrhage_type": "external",
                    "flow_rate_ml_per_min": round(80.0 + 140.0 * severity, 1),
                }
            ]
            if severity >= 0.45:
                steps.append(
                    {
                        "action": "set_hemorrhage",
                        "compartment": "spleen",
                        "hemorrhage_type": "internal",
                        "flow_rate_ml_per_min": round(25.0 + 75.0 * severity, 1),
                    }
                )
            steps.append({"action": "advance_time", "seconds": round(60.0 + 150.0 * severity, 1)})
            return steps
        if injury_type == "cardiac_tamponade":
            return [
                {
                    "action": "set_pericardial_effusion",
                    "effusion_rate_ml_per_min": round(35.0 + 135.0 * severity, 1),
                },
                {"action": "advance_time", "seconds": round(60.0 + 120.0 * severity, 1)},
            ]
        return [
            {"action": "set_tension_pneumothorax", "side": "left", "severity": round(0.30 + 0.50 * severity, 3)},
            {
                "action": "set_hemorrhage",
                "compartment": "right_leg",
                "hemorrhage_type": "external",
                "flow_rate_ml_per_min": round(90.0 + 120.0 * severity, 1),
            },
            {
                "action": "set_hemorrhage",
                "compartment": "spleen",
                "hemorrhage_type": "internal",
                "flow_rate_ml_per_min": round(20.0 + 85.0 * severity, 1),
            },
            {"action": "advance_time", "seconds": round(75.0 + 150.0 * severity, 1)},
        ]
