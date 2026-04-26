"""Shared runtime effects for observation noise and episode time pressure."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from .patient_state import PatientState


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


@dataclass(frozen=True)
class ObservationNoiseConfig:
    """Configures noisy and partially observed bedside monitor readings."""

    noise_level: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.noise_level > 0.0

    @property
    def normalized_level(self) -> float:
        return _clip(float(self.noise_level), 0.0, 1.0)


class NoisyObservation:
    """Applies configurable noise and dropouts to observed patient state."""

    def __init__(self, noise_level: float = 0.0) -> None:
        self._config = ObservationNoiseConfig(noise_level=noise_level)

    @property
    def config(self) -> ObservationNoiseConfig:
        return self._config

    def apply(
        self,
        state: PatientState,
        *,
        rng: random.Random,
    ) -> tuple[PatientState, dict[str, Any]]:
        if not self._config.enabled:
            return state.model_copy(deep=True), {
                "enabled": False,
                "noise_level": 0.0,
                "masked_fields": [],
                "perturbed_fields": [],
            }

        level = self._config.normalized_level
        updates = state.model_dump()
        masked_fields: list[str] = []
        perturbed_fields: list[str] = []

        self._perturb_numeric(updates, "heart_rate_bpm", rng, std_dev=5.0 * level, lower=0.0, upper=240.0, perturbed_fields=perturbed_fields)
        self._perturb_numeric(updates, "systolic_bp_mmhg", rng, std_dev=4.0 * level, lower=40.0, upper=260.0, perturbed_fields=perturbed_fields)
        self._perturb_numeric(updates, "diastolic_bp_mmhg", rng, std_dev=3.0 * level, lower=20.0, upper=180.0, perturbed_fields=perturbed_fields)
        self._perturb_numeric(updates, "spo2", rng, std_dev=0.02 * level, lower=0.5, upper=1.0, perturbed_fields=perturbed_fields)
        self._perturb_numeric(updates, "respiration_rate_bpm", rng, std_dev=2.0 * level, lower=4.0, upper=60.0, perturbed_fields=perturbed_fields)
        self._perturb_numeric(updates, "etco2_mmhg", rng, std_dev=2.5 * level, lower=5.0, upper=90.0, perturbed_fields=perturbed_fields)
        self._perturb_numeric(updates, "core_temperature_c", rng, std_dev=0.2 * level, lower=30.0, upper=43.0, perturbed_fields=perturbed_fields)

        if rng.random() < 0.05 * level:
            updates["spo2"] = None
            masked_fields.append("spo2")
        if rng.random() < 0.035 * level:
            updates["systolic_bp_mmhg"] = None
            updates["diastolic_bp_mmhg"] = None
            masked_fields.extend(["systolic_bp_mmhg", "diastolic_bp_mmhg"])
        if rng.random() < 0.03 * level:
            updates["respiration_rate_bpm"] = None
            masked_fields.append("respiration_rate_bpm")
        if rng.random() < 0.02 * level:
            updates["etco2_mmhg"] = None
            masked_fields.append("etco2_mmhg")

        if updates.get("systolic_bp_mmhg") is None or updates.get("diastolic_bp_mmhg") is None:
            updates["mean_arterial_pressure_mmhg"] = None
            updates["shock_index"] = None
        else:
            systolic = float(updates["systolic_bp_mmhg"])
            diastolic = float(updates["diastolic_bp_mmhg"])
            updates["mean_arterial_pressure_mmhg"] = round((systolic + 2.0 * diastolic) / 3.0, 3)
            if updates.get("heart_rate_bpm") not in (None, 0):
                updates["shock_index"] = round(float(updates["heart_rate_bpm"]) / systolic, 3)

        if rng.random() < 0.02 * level:
            updates["breath_sounds"] = "unclear"
            perturbed_fields.append("breath_sounds")

        observed_state = PatientState(**updates)
        metadata = {
            "enabled": True,
            "noise_level": round(level, 3),
            "masked_fields": sorted(set(masked_fields)),
            "perturbed_fields": sorted(set(perturbed_fields)),
        }
        return observed_state, metadata

    @staticmethod
    def _perturb_numeric(
        updates: dict[str, Any],
        field_name: str,
        rng: random.Random,
        *,
        std_dev: float,
        lower: float,
        upper: float,
        perturbed_fields: list[str],
    ) -> None:
        current_value = updates.get(field_name)
        if current_value is None or std_dev <= 0.0:
            return
        noisy_value = _clip(float(current_value) + rng.gauss(0.0, std_dev), lower, upper)
        updates[field_name] = round(noisy_value, 3)
        perturbed_fields.append(field_name)


@dataclass(frozen=True)
class TimePressureConfig:
    """Configures the urgency curve for delayed trauma intervention."""

    enabled: bool = False
    onset_s: float = 180.0
    escalation_per_minute: float = 0.15
    min_intervention_effectiveness: float = 0.45


class TimePressureMechanic:
    """Computes time-pressure multipliers for delayed trauma management."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        onset_s: float = 180.0,
        escalation_per_minute: float = 0.15,
        min_intervention_effectiveness: float = 0.45,
    ) -> None:
        self._config = TimePressureConfig(
            enabled=bool(enabled),
            onset_s=float(onset_s),
            escalation_per_minute=float(escalation_per_minute),
            min_intervention_effectiveness=float(min_intervention_effectiveness),
        )

    @property
    def config(self) -> TimePressureConfig:
        return self._config

    def deterioration_multiplier(
        self,
        *,
        sim_time_s: float,
        injury_severity: float,
        unstable: bool,
    ) -> float:
        if not self._config.enabled or not unstable or sim_time_s < self._config.onset_s:
            return 1.0

        severity = _clip(float(injury_severity), 0.0, 1.0)
        excess_seconds = max(0.0, float(sim_time_s) - self._config.onset_s)
        return 1.0 + (excess_seconds / 60.0) * self._config.escalation_per_minute * severity

    def intervention_effectiveness_multiplier(
        self,
        *,
        sim_time_s: float,
        injury_severity: float,
        unstable: bool,
    ) -> float:
        deterioration = self.deterioration_multiplier(
            sim_time_s=sim_time_s,
            injury_severity=injury_severity,
            unstable=unstable,
        )
        if deterioration <= 1.0:
            return 1.0

        loss = (deterioration - 1.0) * 0.5
        return max(self._config.min_intervention_effectiveness, 1.0 - loss)

    def as_metadata(
        self,
        *,
        sim_time_s: float,
        injury_severity: float,
        unstable: bool,
    ) -> dict[str, Any]:
        return {
            "enabled": self._config.enabled,
            "onset_s": self._config.onset_s,
            "escalation_per_minute": self._config.escalation_per_minute,
            "injury_severity": round(_clip(float(injury_severity), 0.0, 1.0), 3),
            "deterioration_multiplier": round(
                self.deterioration_multiplier(
                    sim_time_s=sim_time_s,
                    injury_severity=injury_severity,
                    unstable=unstable,
                ),
                3,
            ),
            "intervention_effectiveness_multiplier": round(
                self.intervention_effectiveness_multiplier(
                    sim_time_s=sim_time_s,
                    injury_severity=injury_severity,
                    unstable=unstable,
                ),
                3,
            ),
        }
