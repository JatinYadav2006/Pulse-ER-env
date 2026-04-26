"""Injury-stacking adversary evaluation for generated trauma combinations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .episode_runner import EpisodeRunner, EpisodeTrace
from .policies import Policy, build_expert_policy
from .real_backend import RealPulseBackend
from .server.pathology_architect import (
    DEFAULT_STACKED_INJURY_COMBOS,
    PathologyArchitect,
    PathologyBlueprint,
)


def _map_value(trace: EpisodeTrace) -> float | None:
    observation = trace.final_observation
    if observation.mean_arterial_pressure_mmhg is not None:
        return observation.mean_arterial_pressure_mmhg
    if observation.systolic_bp_mmhg is None or observation.diastolic_bp_mmhg is None:
        return None
    return (observation.systolic_bp_mmhg + 2 * observation.diastolic_bp_mmhg) / 3.0


@dataclass(frozen=True)
class InjuryStackEpisodeResult:
    """Outcome for one patient/combo adversary episode."""

    patient_id: str
    injury_combo: tuple[str, ...]
    severity: float
    requested_severity: float
    reset_adjusted: bool
    reset_terminal: bool
    scenario_id: str
    total_reward: float
    termination_reason: str
    survived: bool
    failed: bool
    num_steps: int
    sim_time_s: float
    final_spo2_percent: float | None
    final_map_mmhg: float | None
    active_alerts: tuple[str, ...]
    events: tuple[str, ...]
    pathology_blueprint: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "injury_combo": list(self.injury_combo),
            "severity": self.severity,
            "requested_severity": self.requested_severity,
            "reset_adjusted": self.reset_adjusted,
            "reset_terminal": self.reset_terminal,
            "scenario_id": self.scenario_id,
            "total_reward": self.total_reward,
            "termination_reason": self.termination_reason,
            "survived": self.survived,
            "failed": self.failed,
            "num_steps": self.num_steps,
            "sim_time_s": self.sim_time_s,
            "final_spo2_percent": self.final_spo2_percent,
            "final_map_mmhg": self.final_map_mmhg,
            "active_alerts": list(self.active_alerts),
            "events": list(self.events),
            "pathology_blueprint": self.pathology_blueprint,
        }


@dataclass(frozen=True)
class PatientAdversaryResult:
    """Adversary summary for one patient across all injury combos."""

    patient_id: str
    severity: float
    results: tuple[InjuryStackEpisodeResult, ...]
    breaking_combo: tuple[str, ...] | None
    breaking_severity: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "severity": self.severity,
            "breaking_combo": list(self.breaking_combo) if self.breaking_combo is not None else None,
            "breaking_severity": self.breaking_severity,
            "breaking_point": (
                {
                    "injury_combo": list(self.breaking_combo),
                    "severity": self.breaking_severity,
                }
                if self.breaking_combo is not None and self.breaking_severity is not None
                else None
            ),
            "results": [result.as_dict() for result in self.results],
        }


class _GeneratedBlueprintBackend(RealPulseBackend):
    """Real backend wrapper that always resets into one generated blueprint."""

    def __init__(
        self,
        blueprint: PathologyBlueprint,
        *,
        environment_factory: Callable[[], object] | None = None,
    ) -> None:
        self._blueprint = blueprint
        super().__init__(default_scenario_id=blueprint.scenario_id, environment_factory=environment_factory)

    def reset(self, scenario_id: str | None = None):
        del scenario_id
        observation = self._environment.reset(pathology_blueprint=self._blueprint.as_dict())
        return self._wrap_observation(observation)


class InjuryStackAdversary:
    """Evaluate how stacked injury combinations affect an agent's robustness."""

    RESET_SURVIVAL_SEVERITY_FLOOR = 0.5
    RESET_SURVIVAL_SEVERITY_STEP = 0.1

    def __init__(
        self,
        *,
        architect: PathologyArchitect | None = None,
        policy_factory: Callable[[], Policy] = build_expert_policy,
        injury_combos: Sequence[Sequence[str]] = DEFAULT_STACKED_INJURY_COMBOS,
        severity: float = 0.7,
        max_steps: int = 12,
        environment_factory: Callable[[], object] | None = None,
    ) -> None:
        self._architect = architect or PathologyArchitect()
        self._policy_factory = policy_factory
        self._injury_combos = tuple(tuple(str(item) for item in combo) for combo in injury_combos)
        self._severity = severity
        self._max_steps = max_steps
        self._environment_factory = environment_factory

    @property
    def injury_combos(self) -> tuple[tuple[str, ...], ...]:
        return self._injury_combos

    def evaluate_combo(
        self,
        *,
        patient_id: str,
        injury_combo: Sequence[str],
        severity: float | None = None,
    ) -> InjuryStackEpisodeResult:
        """Run one generated stacked-injury episode and capture the outcome."""

        requested_severity = self._severity if severity is None else float(severity)
        effective_severity, blueprint, trace, init_error = self._run_with_reset_backoff(
            patient_id=patient_id,
            injury_combo=injury_combo,
            requested_severity=requested_severity,
        )

        if trace is None:
            return InjuryStackEpisodeResult(
                patient_id=patient_id,
                injury_combo=tuple(blueprint.injury_types),
                severity=effective_severity,
                requested_severity=requested_severity,
                reset_adjusted=effective_severity != requested_severity,
                reset_terminal=True,
                scenario_id=blueprint.scenario_id,
                total_reward=-30.0,
                termination_reason="initialization_error",
                survived=False,
                failed=True,
                num_steps=0,
                sim_time_s=0.0,
                final_spo2_percent=None,
                final_map_mmhg=None,
                active_alerts=(),
                events=(f"Generated-case execution failed: {init_error}",),
                pathology_blueprint=blueprint.as_dict(),
            )

        survived = trace.termination_reason.value != "patient_death"
        failed = (not survived) or trace.total_reward <= 0.0
        return InjuryStackEpisodeResult(
            patient_id=patient_id,
            injury_combo=tuple(blueprint.injury_types),
            severity=effective_severity,
            requested_severity=requested_severity,
            reset_adjusted=effective_severity != requested_severity,
            reset_terminal=trace.num_steps == 0 and trace.termination_reason.value == "patient_death",
            scenario_id=trace.scenario_id,
            total_reward=trace.total_reward,
            termination_reason=trace.termination_reason.value,
            survived=survived,
            failed=failed,
            num_steps=trace.num_steps,
            sim_time_s=trace.final_observation.sim_time_s,
            final_spo2_percent=round(trace.final_observation.spo2 * 100.0, 1)
            if trace.final_observation.spo2 is not None
            else None,
            final_map_mmhg=round(_map_value(trace), 1) if _map_value(trace) is not None else None,
            active_alerts=tuple(trace.final_observation.active_alerts),
            events=tuple(trace.events),
            pathology_blueprint=blueprint.as_dict(),
        )

    def find_breaking_combo(
        self,
        patient_id: str,
        *,
        severity: float | None = None,
    ) -> PatientAdversaryResult:
        """Run the default combo ladder for one patient and find the first failing combo."""

        effective_severity = self._severity if severity is None else float(severity)
        results = tuple(
            self.evaluate_combo(
                patient_id=patient_id,
                injury_combo=combo,
                severity=effective_severity,
            )
            for combo in self._injury_combos
        )
        breaking_result = next((result for result in results if result.failed), None)
        breaking_combo = breaking_result.injury_combo if breaking_result is not None else None
        return PatientAdversaryResult(
            patient_id=patient_id,
            severity=effective_severity,
            results=results,
            breaking_combo=breaking_combo,
            breaking_severity=breaking_result.severity if breaking_result is not None else None,
        )

    def run_population(
        self,
        *,
        patient_ids: Sequence[str] | None = None,
        severity: float | None = None,
    ) -> tuple[PatientAdversaryResult, ...]:
        """Run the combo ladder across a patient cohort."""

        selected_patients = (
            list(patient_ids)
            if patient_ids is not None
            else self._architect.supported_patients()
        )
        return tuple(
            self.find_breaking_combo(patient_id=patient_id, severity=severity)
            for patient_id in selected_patients
        )

    def _run_with_reset_backoff(
        self,
        *,
        patient_id: str,
        injury_combo: Sequence[str],
        requested_severity: float,
    ) -> tuple[float, PathologyBlueprint, EpisodeTrace | None, Exception | None]:
        """Retry reset-terminal combos at lower severity so they do not poison the map."""

        attempted_severity = round(float(requested_severity), 2)
        last_error: Exception | None = None
        last_blueprint: PathologyBlueprint | None = None

        while attempted_severity >= self.RESET_SURVIVAL_SEVERITY_FLOOR - 1e-9:
            blueprint = self._architect.build_blueprint(
                patient_id=patient_id,
                injury_types=list(injury_combo),
                severity=attempted_severity,
            )
            last_blueprint = blueprint
            backend = _GeneratedBlueprintBackend(
                blueprint,
                environment_factory=self._environment_factory,
            )
            runner = EpisodeRunner(backend=backend, max_steps=self._max_steps)
            policy = self._policy_factory()
            try:
                trace = runner.run(policy=policy, scenario_id=blueprint.scenario_id)
            except Exception as exc:
                last_error = exc
                if attempted_severity <= self.RESET_SURVIVAL_SEVERITY_FLOOR + 1e-9:
                    return attempted_severity, blueprint, None, exc
                attempted_severity = round(attempted_severity - self.RESET_SURVIVAL_SEVERITY_STEP, 2)
                continue
            finally:
                backend.close()

            if not self._is_reset_terminal(trace):
                return attempted_severity, blueprint, trace, None

            if attempted_severity <= self.RESET_SURVIVAL_SEVERITY_FLOOR + 1e-9:
                return attempted_severity, blueprint, trace, None
            attempted_severity = round(attempted_severity - self.RESET_SURVIVAL_SEVERITY_STEP, 2)

        assert last_blueprint is not None
        return self.RESET_SURVIVAL_SEVERITY_FLOOR, last_blueprint, None, last_error

    @staticmethod
    def _is_reset_terminal(trace: EpisodeTrace) -> bool:
        """Return whether the episode ended before any action because reset was already terminal."""

        return trace.num_steps == 0 and trace.termination_reason.value == "patient_death"


def main() -> None:
    """CLI entry point for quick adversary runs and JSON output."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", action="append", dest="patient_ids", help="Patient id to evaluate. Repeat for more than one.")
    parser.add_argument("--severity", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--json", action="store_true", help="Emit structured JSON instead of a text summary.")
    args = parser.parse_args()

    adversary = InjuryStackAdversary(
        severity=args.severity,
        max_steps=args.max_steps,
    )
    results = adversary.run_population(patient_ids=args.patient_ids, severity=args.severity)

    if args.json:
        print(json.dumps([result.as_dict() for result in results], indent=2))
        return

    print("Injury stack adversary")
    for patient_result in results:
        print(f"\n{patient_result.patient_id}")
        print(f"  breaking_combo: {patient_result.breaking_combo or 'none'}")
        for result in patient_result.results:
            combo_label = " + ".join(result.injury_combo)
            print(
                f"  combo={combo_label}"
                f" reward={result.total_reward:.3f}"
                f" termination={result.termination_reason}"
                f" survived={result.survived}"
            )


if __name__ == "__main__":
    main()
