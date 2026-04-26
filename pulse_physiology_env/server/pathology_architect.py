"""Scenario authoring for generated trauma cases and stacked-injury adversaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

from pulse_physiology_env.patient_state import ScenarioDifficulty

from .pulse_engine_adapter import PulseEngineAdapter
from .scenarios import (
    EASY_PATIENTS,
    HARD_PATIENTS,
    MEDIUM_PATIENTS,
    PatientProfile,
    ScenarioDefinition,
)

AtomicInjuryType = Literal["tension_pneumothorax", "hemorrhagic_shock", "cardiac_tamponade"]
InjuryType = Literal["tension_pneumothorax", "hemorrhagic_shock", "cardiac_tamponade", "polytrauma"]

DEFAULT_STACKED_INJURY_COMBOS: tuple[tuple[AtomicInjuryType, ...], ...] = (
    ("tension_pneumothorax",),
    ("hemorrhagic_shock",),
    ("cardiac_tamponade",),
    ("tension_pneumothorax", "hemorrhagic_shock"),
    ("hemorrhagic_shock", "cardiac_tamponade"),
    ("tension_pneumothorax", "hemorrhagic_shock", "cardiac_tamponade"),
)


@dataclass(frozen=True)
class PathologyBlueprint:
    """Generated case definition that can be applied directly to Pulse."""

    scenario_id: str
    description: str
    patient_id: str
    state_file: str
    injury_type: InjuryType
    injury_types: tuple[AtomicInjuryType, ...]
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

    @staticmethod
    def default_injury_combos() -> list[list[str]]:
        """Return the default stacked-injury combos used by the adversary runner."""

        return [list(combo) for combo in DEFAULT_STACKED_INJURY_COMBOS]

    def build_blueprint(
        self,
        *,
        patient_id: str,
        injury_type: str | None = None,
        injury_types: Sequence[str] | None = None,
        severity: float,
    ) -> PathologyBlueprint:
        """Build a generated case from one or more injury types and a severity value."""

        profile = self._resolve_patient(patient_id)
        combo = self._resolve_injury_combo(injury_type=injury_type, injury_types=injury_types)
        clamped_severity = max(0.0, min(1.0, float(severity)))
        difficulty = self._difficulty_from_severity(clamped_severity)
        summary_injury = combo[0] if len(combo) == 1 else "polytrauma"
        reward_profile = summary_injury
        setup_actions = self._build_setup_actions(combo, clamped_severity)
        combo_slug = "_plus_".join(combo)
        scenario_id = f"generated_{combo_slug}_{profile.patient_id}"
        combo_label = " + ".join(injury.replace("_", " ") for injury in combo)
        description = (
            f"Generated {combo_label} case for {profile.patient_id} at severity {clamped_severity:.2f}."
        )
        max_time_s = 1800.0 + (300.0 if len(combo) >= 2 else 0.0) + (300.0 if len(combo) >= 3 else 0.0)
        return PathologyBlueprint(
            scenario_id=scenario_id,
            description=description,
            patient_id=profile.patient_id,
            state_file=profile.state_file,
            injury_type=summary_injury,  # type: ignore[arg-type]
            injury_types=combo,
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
                self._advance_setup_time(adapter, float(step["seconds"]))
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

    def _resolve_injury_combo(
        self,
        *,
        injury_type: str | None,
        injury_types: Sequence[str] | None,
    ) -> tuple[AtomicInjuryType, ...]:
        if injury_type is not None and injury_types is not None:
            raise ValueError("Pass either injury_type or injury_types, not both.")
        if injury_type is None and injury_types is None:
            raise ValueError("Pass injury_type or injury_types when generating a pathology blueprint.")

        raw_items: list[str]
        if injury_types is not None:
            if not injury_types:
                raise ValueError("injury_types must contain at least one injury.")
            raw_items = [str(item) for item in injury_types]
        else:
            raw_items = [str(injury_type)]

        combo: list[AtomicInjuryType] = []
        for raw_item in raw_items:
            injury_key = raw_item.strip().lower().replace("-", "_").replace(" ", "_")
            if injury_key == "polytrauma":
                for poly_injury in ("tension_pneumothorax", "hemorrhagic_shock"):
                    if poly_injury not in combo:
                        combo.append(poly_injury)
                continue
            if injury_key not in {"tension_pneumothorax", "hemorrhagic_shock", "cardiac_tamponade"}:
                valid = ", ".join(self.supported_injury_types())
                raise ValueError(f"Unsupported injury type '{raw_item}'. Expected one of: {valid}")
            typed_injury = injury_key  # type: ignore[assignment]
            if typed_injury not in combo:
                combo.append(typed_injury)

        return tuple(combo)

    @staticmethod
    def _difficulty_from_severity(severity: float) -> ScenarioDifficulty:
        if severity < 0.34:
            return "easy"
        if severity < 0.67:
            return "medium"
        return "hard"

    def _build_setup_actions(
        self,
        injury_types: tuple[AtomicInjuryType, ...],
        severity: float,
    ) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []
        combo_size = len(injury_types)
        for injury_type in injury_types:
            steps.extend(self._build_atomic_injury_actions(injury_type, severity, combo_size=combo_size))
        steps.append(
            {
                "action": "advance_time",
                "seconds": self._initial_deterioration_seconds(injury_types, severity, combo_size=combo_size),
            }
        )
        return steps

    def _build_atomic_injury_actions(
        self,
        injury_type: AtomicInjuryType,
        severity: float,
        *,
        combo_size: int,
    ) -> list[dict[str, Any]]:
        combo_scale = self._combo_intensity_scale(combo_size)
        if injury_type == "tension_pneumothorax":
            return [
                {
                    "action": "set_tension_pneumothorax",
                    "side": "left",
                    "severity": round((0.30 + 0.55 * severity) * combo_scale, 3),
                }
            ]
        if injury_type == "hemorrhagic_shock":
            steps: list[dict[str, Any]] = [
                {
                    "action": "set_hemorrhage",
                    "compartment": "right_leg",
                    "hemorrhage_type": "external",
                    "flow_rate_ml_per_min": round((80.0 + 140.0 * severity) * combo_scale, 1),
                }
            ]
            if severity >= 0.45:
                steps.append(
                    {
                        "action": "set_hemorrhage",
                        "compartment": "spleen",
                        "hemorrhage_type": "internal",
                        "flow_rate_ml_per_min": round((25.0 + 75.0 * severity) * combo_scale, 1),
                    }
                )
            return steps
        return [
            {
                "action": "set_pericardial_effusion",
                "effusion_rate_ml_per_min": round((35.0 + 135.0 * severity) * combo_scale, 1),
            }
        ]

    @staticmethod
    def _initial_deterioration_seconds(
        injury_types: tuple[AtomicInjuryType, ...],
        severity: float,
        *,
        combo_size: int,
    ) -> float:
        base = 45.0 + 135.0 * severity
        if "hemorrhagic_shock" in injury_types:
            base = max(base, 60.0 + 150.0 * severity)
        if "cardiac_tamponade" in injury_types:
            base = max(base, 60.0 + 120.0 * severity)
        combo_time_scale = {1: 1.0, 2: 0.78, 3: 0.62}.get(combo_size, 0.55)
        return round(min(240.0, base * combo_time_scale), 1)

    @staticmethod
    def _combo_intensity_scale(combo_size: int) -> float:
        return {1: 1.0, 2: 0.84, 3: 0.72}.get(combo_size, 0.68)

    @staticmethod
    def _advance_setup_time(adapter: PulseEngineAdapter, total_seconds: float) -> None:
        """Advance generated scenarios in smaller chunks so severe stacks reset reliably."""

        remaining_seconds = max(0.0, float(total_seconds))
        chunk_seconds = min(30.0, remaining_seconds)
        while remaining_seconds > 0.0:
            step_seconds = min(chunk_seconds, remaining_seconds)
            try:
                state = adapter.advance_time(step_seconds)
                remaining_seconds = round(max(0.0, remaining_seconds - step_seconds), 6)
                if state.done or "cardiac_arrest" in state.active_alerts:
                    return
            except RuntimeError:
                if step_seconds <= 5.0:
                    return
                chunk_seconds = max(5.0, step_seconds / 2.0)
