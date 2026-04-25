"""Server-side adapter around the locally built Pulse physiology engine."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from pulse_physiology_env.patient_state import (
    ArterialBloodGasResult,
    BasicMetabolicPanelResult,
    CompleteBloodCountResult,
    PatientState,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INSTALL_DIR = _REPO_ROOT / "engine-build" / "install"


def _bootstrap_pulse_paths(explicit_install_dir: Path | None = None) -> Path:
    candidates: list[Path] = []

    if explicit_install_dir is not None:
        candidates.append(Path(explicit_install_dir))

    env_install_dir = os.getenv("PULSE_INSTALL_DIR")
    if env_install_dir:
        candidates.append(Path(env_install_dir))

    candidates.append(_DEFAULT_INSTALL_DIR)

    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        checked.append(str(resolved))
        bin_dir = resolved / "bin"
        python_dir = resolved / "python"
        if not (bin_dir.exists() and python_dir.exists()):
            continue

        for path in (str(bin_dir), str(python_dir)):
            if path not in sys.path:
                sys.path.insert(0, path)
        return resolved

    raise RuntimeError(
        "Could not locate a Pulse install with both 'bin' and 'python' folders. "
        f"Checked: {', '.join(checked)}"
    )


_PULSE_IMPORT_ERROR: Exception | None = None
_BOOTSTRAP_INSTALL_DIR: Path | None = None
try:
    _BOOTSTRAP_INSTALL_DIR = _bootstrap_pulse_paths()

    from pulse.engine.PulseEngine import PulseEngine
    from pulse.cdm.engine import SEAction, SEDataRequest, SEDataRequestManager, eSide, eSwitch
    from pulse.cdm.patient_actions import (
        SEHemorrhage,
        SEIntubation,
        SENeedleDecompression,
        SESubstanceInfusion,
        SETensionPneumothorax,
        eIntubationType,
    )
    from pulse.cdm.scalars import (
        AmountPerVolumeUnit,
        FrequencyUnit,
        MassPerVolumeUnit,
        MassUnit,
        PressureTimePerVolumeUnit,
        PressureUnit,
        TemperatureUnit,
        VolumePerPressureUnit,
        VolumePerTimeUnit,
        VolumeUnit,
    )
except Exception as exc:  # pragma: no cover - exercised only on misconfigured hosts
    _PULSE_IMPORT_ERROR = exc


@dataclass
class _RuntimeContext:
    state_file: Path | None = None
    action_budget_remaining: int | None = None
    pending_diagnostics: dict[str, int] = field(default_factory=dict)
    active_infusions: dict[str, float] = field(default_factory=dict)
    active_hemorrhages: dict[str, float] = field(default_factory=dict)
    active_tension_pneumothorax_sides: set[str] = field(default_factory=set)
    pain_sources: dict[str, float] = field(default_factory=dict)
    intubation_type: str = "Off"


class PulseEngineAdapter:
    """Loads Pulse states, advances simulation time, and synthesizes PatientState."""

    DEFAULT_STATE_FILENAME = "StandardMale@0s.json"
    _SODIUM_MOLAR_MASS = 22.98976928
    _POTASSIUM_MOLAR_MASS = 39.0983
    _CALCIUM_MOLAR_MASS = 40.078

    def __init__(
        self,
        install_dir: str | Path | None = None,
        default_state_file: str | Path | None = None,
        *,
        log_to_console: bool = False,
        action_budget_remaining: int | None = None,
    ) -> None:
        if _PULSE_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Pulse Python bindings could not be imported. "
                "Make sure the local Pulse build exists under engine-build/install "
                "or set PULSE_INSTALL_DIR."
            ) from _PULSE_IMPORT_ERROR

        self._install_dir = _bootstrap_pulse_paths(
            Path(install_dir) if install_dir is not None else _BOOTSTRAP_INSTALL_DIR
        )
        self._bin_dir = self._install_dir / "bin"
        self._states_dir = self._bin_dir / "states"
        self._default_state_file = self._resolve_state_file(default_state_file)
        self._log_to_console = log_to_console
        self._default_action_budget_remaining = action_budget_remaining

        self._engine: PulseEngine | None = None
        self._data_request_keys, requests = self._build_data_requests()
        self._data_request_mgr = SEDataRequestManager(requests)
        self._latest_raw_metrics: dict[str, float | None] = {}
        self._last_lactate_mg_per_dl: float | None = None
        self._runtime = _RuntimeContext(action_budget_remaining=action_budget_remaining)

    def load_patient(
        self,
        state_file: str | Path | None = None,
        *,
        initial_actions: Sequence[SEAction] | None = None,
        action_budget_remaining: int | None = None,
    ) -> PatientState:
        """Load a baseline Pulse state into a fresh engine instance."""

        resolved_state_file = (
            self._default_state_file if state_file is None else self._resolve_state_file(state_file)
        )
        self.close()

        self._engine = PulseEngine(data_root_dir=str(self._bin_dir))
        self._engine.log_to_console(self._log_to_console)

        if not self._engine.serialize_from_file(str(resolved_state_file), self._data_request_mgr):
            raise RuntimeError(f"Pulse failed to load state file: {resolved_state_file}")

        self._runtime = _RuntimeContext(
            state_file=resolved_state_file,
            action_budget_remaining=(
                self._default_action_budget_remaining
                if action_budget_remaining is None
                else action_budget_remaining
            ),
        )
        self._refresh_raw_metrics()
        self._last_lactate_mg_per_dl = self._latest_raw_metrics.get("lactate_mg_per_dl")

        if initial_actions:
            return self.apply_actions(initial_actions)

        return self.get_full_state()

    def apply_actions(
        self,
        actions: Sequence[SEAction] | Iterable[SEAction],
        *,
        advance_time_seconds: float | None = None,
    ) -> PatientState:
        """Submit native Pulse actions and optionally advance the sim immediately."""

        engine = self._require_engine()
        action_list = list(actions)
        if not action_list:
            return self.get_full_state()

        engine.process_actions(action_list)
        for action in action_list:
            self._track_runtime_action(action)

        if advance_time_seconds is not None and advance_time_seconds > 0:
            return self.advance_time(advance_time_seconds)

        self._refresh_raw_metrics()
        return self.get_full_state()

    def advance_time(self, seconds: float) -> PatientState:
        """Advance the simulation clock and return the updated state."""

        if seconds < 0:
            raise ValueError("Simulation time cannot be advanced by a negative duration.")

        engine = self._require_engine()
        if not engine.advance_time_s(seconds):
            raise RuntimeError(f"Pulse failed to advance by {seconds} simulated seconds.")

        self._refresh_raw_metrics()
        return self.get_full_state()

    def get_full_state(self) -> PatientState:
        """Return the current typed patient state."""

        self._require_engine()
        if not self._latest_raw_metrics:
            self._refresh_raw_metrics()

        metrics = self._latest_raw_metrics
        blood_volume_ml = metrics.get("blood_volume_ml")
        hemoglobin_content_g = metrics.get("hemoglobin_content_g")
        blood_volume_dl = blood_volume_ml / 100.0 if blood_volume_ml and blood_volume_ml > 0 else None
        hemoglobin_g_per_dl = (
            hemoglobin_content_g / blood_volume_dl
            if hemoglobin_content_g is not None and blood_volume_dl not in (None, 0)
            else None
        )

        arterial_pco2 = metrics.get("arterial_carbon_dioxide_pressure_mm_hg")
        blood_ph = metrics.get("blood_ph")
        bicarbonate = self._derive_bicarbonate_meq_per_l(blood_ph, arterial_pco2)
        base_excess = metrics.get("base_excess_meq_per_l")
        base_deficit = max(-base_excess, 0.0) if base_excess is not None else None

        abg_result = ArterialBloodGasResult(
            ph=blood_ph,
            partial_pressure_of_oxygen_mm_hg=metrics.get("arterial_oxygen_pressure_mm_hg"),
            partial_pressure_of_carbon_dioxide_mm_hg=arterial_pco2,
            oxygen_saturation=metrics.get("oxygen_saturation"),
            bicarbonate_meq_per_l=bicarbonate,
            lactate_mg_per_dl=metrics.get("lactate_mg_per_dl"),
            base_excess_meq_per_l=base_excess,
            base_deficit_meq_per_l=base_deficit,
        )
        cbc_result = CompleteBloodCountResult(
            hemoglobin_g_per_dl=hemoglobin_g_per_dl,
            hematocrit_fraction=metrics.get("hematocrit_fraction"),
            white_blood_cell_count_per_u_l=metrics.get("white_blood_cell_count_per_u_l"),
            platelet_count_per_u_l=None,
            red_blood_cell_count_per_u_l=metrics.get("red_blood_cell_count_per_u_l"),
        )
        bmp_result = BasicMetabolicPanelResult(
            sodium_mmol_per_l=self._convert_mg_per_dl_to_mmol_per_l(
                metrics.get("sodium_mg_per_dl"),
                self._SODIUM_MOLAR_MASS,
            ),
            potassium_mmol_per_l=self._convert_mg_per_dl_to_mmol_per_l(
                metrics.get("potassium_mg_per_dl"),
                self._POTASSIUM_MOLAR_MASS,
            ),
            calcium_mmol_per_l=self._convert_mg_per_dl_to_mmol_per_l(
                metrics.get("calcium_mg_per_dl"),
                self._CALCIUM_MOLAR_MASS,
            ),
            creatinine_mg_per_dl=metrics.get("creatinine_mg_per_dl"),
            glucose_mg_per_dl=metrics.get("glucose_mg_per_dl"),
        )

        is_alive = self._is_alive_from_metrics(metrics)
        sedation_level = metrics.get("sedation_level") or 0.0
        consciousness_level = 0.0 if not is_alive else self._clamp01(1.0 - sedation_level)
        pain_level = self._derive_pain_level()

        state = PatientState(
            heart_rate_bpm=metrics.get("heart_rate_bpm"),
            systolic_bp_mm_hg=metrics.get("systolic_bp_mm_hg"),
            diastolic_bp_mm_hg=metrics.get("diastolic_bp_mm_hg"),
            mean_arterial_pressure_mm_hg=metrics.get("mean_arterial_pressure_mm_hg"),
            cardiac_output_l_per_min=metrics.get("cardiac_output_l_per_min"),
            spo2=metrics.get("oxygen_saturation"),
            respiratory_rate_bpm=metrics.get("respiration_rate_bpm"),
            etco2_mm_hg=metrics.get("end_tidal_carbon_dioxide_pressure_mm_hg"),
            tidal_volume_ml=metrics.get("tidal_volume_ml"),
            breath_sounds=self._derive_breath_sounds(),
            consciousness_level=consciousness_level,
            pain_level=pain_level,
            core_temperature_c=metrics.get("core_temperature_c"),
            abg_result=abg_result,
            cbc_result=cbc_result,
            bmp_result=bmp_result,
            pending_diagnostics=dict(self._runtime.pending_diagnostics),
            active_infusions=dict(self._runtime.active_infusions),
            active_hemorrhages=dict(self._runtime.active_hemorrhages),
            shock_index=self._derive_shock_index(
                metrics.get("heart_rate_bpm"),
                metrics.get("systolic_bp_mm_hg"),
            ),
            lactate_trend=self._derive_lactate_trend(metrics.get("lactate_mg_per_dl")),
            intubated=self._is_intubated(),
            simulated_time_seconds=metrics.get("simulated_time_seconds") or 0.0,
            action_budget_remaining=self._runtime.action_budget_remaining,
        )

        self._last_lactate_mg_per_dl = metrics.get("lactate_mg_per_dl")
        return state

    def get_raw_metrics(self) -> dict[str, float | None]:
        """Return the latest low-level Pulse metrics used to build PatientState."""

        self._require_engine()
        if not self._latest_raw_metrics:
            self._refresh_raw_metrics()
        return dict(self._latest_raw_metrics)

    def is_patient_alive(self) -> bool:
        """Check whether the patient is still in a survivable state."""

        self._require_engine()
        if not self._latest_raw_metrics:
            self._refresh_raw_metrics()
        return self._is_alive_from_metrics(self._latest_raw_metrics)

    def close(self) -> None:
        """Release the active Pulse engine instance."""

        if self._engine is None:
            return

        try:
            self._engine.clear()
        finally:
            self._engine = None
            self._latest_raw_metrics = {}
            self._last_lactate_mg_per_dl = None

    def __enter__(self) -> "PulseEngineAdapter":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _require_engine(self) -> PulseEngine:
        if self._engine is None:
            raise RuntimeError("Pulse engine is not loaded. Call load_patient() first.")
        return self._engine

    def _resolve_state_file(self, state_file: str | Path | None) -> Path:
        if state_file is None:
            candidate = self._states_dir / self.DEFAULT_STATE_FILENAME
        else:
            raw = Path(state_file)
            candidate = raw if raw.is_absolute() else self._states_dir / raw

        resolved = candidate.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Pulse state file was not found: {resolved}")
        return resolved

    def _build_data_requests(self) -> tuple[list[str], list[Any]]:
        request_specs = [
            ("heart_rate_bpm", SEDataRequest.create_physiology_request("HeartRate", unit=FrequencyUnit.Per_min)),
            ("systolic_bp_mm_hg", SEDataRequest.create_physiology_request("SystolicArterialPressure", unit=PressureUnit.mmHg)),
            ("diastolic_bp_mm_hg", SEDataRequest.create_physiology_request("DiastolicArterialPressure", unit=PressureUnit.mmHg)),
            ("mean_arterial_pressure_mm_hg", SEDataRequest.create_physiology_request("MeanArterialPressure", unit=PressureUnit.mmHg)),
            ("cardiac_output_l_per_min", SEDataRequest.create_physiology_request("CardiacOutput", unit=VolumePerTimeUnit.L_Per_min)),
            ("stroke_volume_ml", SEDataRequest.create_physiology_request("HeartStrokeVolume", unit=VolumeUnit.mL)),
            (
                "systemic_vascular_resistance_mm_hg_min_per_l",
                SEDataRequest.create_physiology_request(
                    "SystemicVascularResistance",
                    unit=PressureTimePerVolumeUnit.mmHg_min_Per_L,
                ),
            ),
            ("oxygen_saturation", SEDataRequest.create_physiology_request("OxygenSaturation")),
            ("respiration_rate_bpm", SEDataRequest.create_physiology_request("RespirationRate", unit=FrequencyUnit.Per_min)),
            (
                "end_tidal_carbon_dioxide_pressure_mm_hg",
                SEDataRequest.create_physiology_request("EndTidalCarbonDioxidePressure", unit=PressureUnit.mmHg),
            ),
            ("tidal_volume_ml", SEDataRequest.create_physiology_request("TidalVolume", unit=VolumeUnit.mL)),
            (
                "total_pulmonary_ventilation_l_per_min",
                SEDataRequest.create_physiology_request("TotalPulmonaryVentilation", unit=VolumePerTimeUnit.L_Per_min),
            ),
            (
                "respiratory_compliance_ml_per_cm_h2o",
                SEDataRequest.create_physiology_request(
                    "RespiratoryCompliance",
                    unit=VolumePerPressureUnit.mL_Per_cmH2O,
                ),
            ),
            (
                "inspiratory_respiratory_resistance_cm_h2o_s_per_l",
                SEDataRequest.create_physiology_request(
                    "InspiratoryRespiratoryResistance",
                    unit=PressureTimePerVolumeUnit.cmH2O_s_Per_L,
                ),
            ),
            (
                "expiratory_respiratory_resistance_cm_h2o_s_per_l",
                SEDataRequest.create_physiology_request(
                    "ExpiratoryRespiratoryResistance",
                    unit=PressureTimePerVolumeUnit.cmH2O_s_Per_L,
                ),
            ),
            ("core_temperature_c", SEDataRequest.create_physiology_request("CoreTemperature", unit=TemperatureUnit.C)),
            ("skin_temperature_c", SEDataRequest.create_physiology_request("SkinTemperature", unit=TemperatureUnit.C)),
            ("peripheral_perfusion_index", SEDataRequest.create_physiology_request("PeripheralPerfusionIndex")),
            ("sedation_level", SEDataRequest.create_physiology_request("SedationLevel")),
            ("neuromuscular_block_level", SEDataRequest.create_physiology_request("NeuromuscularBlockLevel")),
            ("fatigue_level", SEDataRequest.create_physiology_request("FatigueLevel")),
            ("blood_ph", SEDataRequest.create_physiology_request("BloodPH")),
            ("base_excess_meq_per_l", SEDataRequest.create_physiology_request("BaseExcess", unit=AmountPerVolumeUnit.mEq_Per_L)),
            ("hematocrit_fraction", SEDataRequest.create_physiology_request("Hematocrit")),
            ("hemoglobin_content_g", SEDataRequest.create_physiology_request("HemoglobinContent", unit=MassUnit.g)),
            (
                "white_blood_cell_count_per_u_l",
                SEDataRequest.create_physiology_request("WhiteBloodCellCount", unit=AmountPerVolumeUnit.ct_Per_uL),
            ),
            (
                "red_blood_cell_count_per_u_l",
                SEDataRequest.create_physiology_request("RedBloodCellCount", unit=AmountPerVolumeUnit.ct_Per_uL),
            ),
            (
                "arterial_oxygen_pressure_mm_hg",
                SEDataRequest.create_physiology_request("ArterialOxygenPressure", unit=PressureUnit.mmHg),
            ),
            (
                "arterial_carbon_dioxide_pressure_mm_hg",
                SEDataRequest.create_physiology_request("ArterialCarbonDioxidePressure", unit=PressureUnit.mmHg),
            ),
            ("blood_volume_ml", SEDataRequest.create_physiology_request("BloodVolume", unit=VolumeUnit.mL)),
            (
                "urine_production_rate_ml_per_min",
                SEDataRequest.create_physiology_request("UrineProductionRate", unit=VolumePerTimeUnit.mL_Per_min),
            ),
            (
                "total_hemorrhage_rate_ml_per_min",
                SEDataRequest.create_physiology_request("TotalHemorrhageRate", unit=VolumePerTimeUnit.mL_Per_min),
            ),
            (
                "lactate_mg_per_dl",
                SEDataRequest.create_liquid_compartment_substance_request(
                    "Aorta",
                    "Lactate",
                    "Concentration",
                    unit=MassPerVolumeUnit.mg_Per_dL,
                ),
            ),
            (
                "sodium_mg_per_dl",
                SEDataRequest.create_liquid_compartment_substance_request(
                    "Aorta",
                    "Sodium",
                    "Concentration",
                    unit=MassPerVolumeUnit.mg_Per_dL,
                ),
            ),
            (
                "potassium_mg_per_dl",
                SEDataRequest.create_liquid_compartment_substance_request(
                    "Aorta",
                    "Potassium",
                    "Concentration",
                    unit=MassPerVolumeUnit.mg_Per_dL,
                ),
            ),
            (
                "calcium_mg_per_dl",
                SEDataRequest.create_liquid_compartment_substance_request(
                    "Aorta",
                    "Calcium",
                    "Concentration",
                    unit=MassPerVolumeUnit.mg_Per_dL,
                ),
            ),
            (
                "glucose_mg_per_dl",
                SEDataRequest.create_liquid_compartment_substance_request(
                    "Aorta",
                    "Glucose",
                    "Concentration",
                    unit=MassPerVolumeUnit.mg_Per_dL,
                ),
            ),
            (
                "creatinine_mg_per_dl",
                SEDataRequest.create_liquid_compartment_substance_request(
                    "Aorta",
                    "Creatinine",
                    "Concentration",
                    unit=MassPerVolumeUnit.mg_Per_dL,
                ),
            ),
        ]
        keys = [key for key, _ in request_specs]
        requests = [request for _, request in request_specs]
        return keys, requests

    def _refresh_raw_metrics(self) -> None:
        results = self._require_engine().pull_data()
        if results is None or len(results) == 0:
            raise RuntimeError("Pulse returned no data for the configured requests.")

        metrics: dict[str, float | None] = {
            "simulated_time_seconds": self._coerce_optional_float(results[0]),
        }
        for offset, key in enumerate(self._data_request_keys, start=1):
            metrics[key] = self._coerce_optional_float(results[offset])
        self._latest_raw_metrics = metrics

    def _track_runtime_action(self, action: SEAction) -> None:
        if isinstance(action, SEIntubation):
            self._runtime.intubation_type = action.get_type().name
            return

        if isinstance(action, SENeedleDecompression) and action.has_side():
            side = action.get_side().name.lower()
            self._runtime.active_tension_pneumothorax_sides.discard(side)
            self._runtime.pain_sources.pop(f"tension_pneumothorax:{side}", None)
            return

        if isinstance(action, SETensionPneumothorax) and action.has_side():
            side = action.get_side().name.lower()
            severity = self._get_scalar_value(action, "has_severity", "get_severity")
            self._runtime.active_tension_pneumothorax_sides.add(side)
            self._runtime.pain_sources[f"tension_pneumothorax:{side}"] = severity or 0.8
            return

        if isinstance(action, SEHemorrhage) and action.has_compartment():
            compartment = str(action.get_compartment())
            rate_ml_per_min = self._get_scalar_value(
                action,
                "has_flow_rate",
                "get_flow_rate",
                unit=VolumePerTimeUnit.mL_Per_min,
            )
            severity = self._get_scalar_value(action, "has_severity", "get_severity")

            if rate_ml_per_min is not None and rate_ml_per_min > 0:
                self._runtime.active_hemorrhages[compartment] = rate_ml_per_min
            else:
                self._runtime.active_hemorrhages.pop(compartment, None)

            estimated_pain = severity if severity is not None else self._estimate_hemorrhage_severity(rate_ml_per_min)
            self._runtime.pain_sources[f"hemorrhage:{compartment}"] = estimated_pain
            return

        if isinstance(action, SESubstanceInfusion) and action.has_substance():
            substance = action.get_substance()
            rate_ml_per_min = self._get_scalar_value(
                action,
                "has_rate",
                "get_rate",
                unit=VolumePerTimeUnit.mL_Per_min,
            )
            concentration_mg_per_ml = self._get_scalar_value(
                action,
                "has_concentration",
                "get_concentration",
                unit=MassPerVolumeUnit.mg_Per_mL,
            )

            delivered_dose = None
            if rate_ml_per_min is not None:
                delivered_dose = (
                    rate_ml_per_min * concentration_mg_per_ml
                    if concentration_mg_per_ml is not None
                    else rate_ml_per_min
                )

            if delivered_dose is not None and delivered_dose > 0:
                self._runtime.active_infusions[substance] = delivered_dose
            else:
                self._runtime.active_infusions.pop(substance, None)

    def _is_alive_from_metrics(self, metrics: dict[str, float | None]) -> bool:
        if self._has_terminal_event():
            return False

        heart_rate = metrics.get("heart_rate_bpm")
        map_mmhg = metrics.get("mean_arterial_pressure_mm_hg")
        spo2 = metrics.get("oxygen_saturation")
        etco2 = metrics.get("end_tidal_carbon_dioxide_pressure_mm_hg")

        if heart_rate is not None and heart_rate <= 0.1:
            return False
        if map_mmhg is not None and map_mmhg <= 10.0:
            return False
        if spo2 is not None and spo2 <= 0.01 and etco2 is not None and etco2 <= 1.0:
            return False
        return True

    def _has_terminal_event(self) -> bool:
        active_events = self._require_engine().pull_active_events() or {}
        terminal_events = {"CardiacArrest", "IrreversibleState", "Asystole"}
        for event in active_events.keys():
            normalized = self._normalize_event_name(event)
            if normalized in terminal_events:
                return True
        return False

    def _derive_breath_sounds(self) -> str:
        left_absent = False
        right_absent = False

        if self._runtime.intubation_type == eIntubationType.LeftMainstem.name:
            right_absent = True
        elif self._runtime.intubation_type == eIntubationType.RightMainstem.name:
            left_absent = True

        if "left" in self._runtime.active_tension_pneumothorax_sides:
            left_absent = True
        if "right" in self._runtime.active_tension_pneumothorax_sides:
            right_absent = True

        if left_absent and right_absent:
            return "absent bilateral"
        if left_absent:
            return "absent left"
        if right_absent:
            return "absent right"
        return "present bilateral"

    def _is_intubated(self) -> bool:
        return self._runtime.intubation_type in {
            eIntubationType.Esophageal.name,
            eIntubationType.LeftMainstem.name,
            eIntubationType.RightMainstem.name,
            eIntubationType.Tracheal.name,
        }

    def _derive_pain_level(self) -> float:
        if not self._runtime.pain_sources:
            return 0.0
        return self._clamp01(max(self._runtime.pain_sources.values()))

    def _derive_shock_index(self, heart_rate_bpm: float | None, systolic_bp_mm_hg: float | None) -> float | None:
        if heart_rate_bpm is None or systolic_bp_mm_hg is None or systolic_bp_mm_hg <= 0:
            return None
        return heart_rate_bpm / systolic_bp_mm_hg

    def _derive_lactate_trend(self, current_lactate_mg_per_dl: float | None) -> str:
        previous = self._last_lactate_mg_per_dl
        if current_lactate_mg_per_dl is None or previous is None:
            return "stable"

        delta = current_lactate_mg_per_dl - previous
        if abs(delta) < 0.25:
            return "stable"
        return "improving" if delta < 0 else "worsening"

    @staticmethod
    def _derive_bicarbonate_meq_per_l(ph: float | None, arterial_pco2_mm_hg: float | None) -> float | None:
        if ph is None or arterial_pco2_mm_hg is None:
            return None
        return 0.03 * arterial_pco2_mm_hg * (10 ** (ph - 6.1))

    @staticmethod
    def _convert_mg_per_dl_to_mmol_per_l(value_mg_per_dl: float | None, molar_mass_g_per_mol: float) -> float | None:
        if value_mg_per_dl is None:
            return None
        return (value_mg_per_dl * 10.0) / molar_mass_g_per_mol

    @staticmethod
    def _estimate_hemorrhage_severity(rate_ml_per_min: float | None) -> float:
        if rate_ml_per_min is None:
            return 0.0
        return max(0.0, min(rate_ml_per_min / 250.0, 1.0))

    @staticmethod
    def _coerce_optional_float(value: Any) -> float | None:
        if value is None:
            return None
        coerced = float(value)
        return None if math.isnan(coerced) else coerced

    @staticmethod
    def _normalize_event_name(event: Any) -> str:
        name = getattr(event, "name", str(event))
        return name.split(".")[-1]

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(value, 1.0))

    @staticmethod
    def _get_scalar_value(
        action: Any,
        has_method_name: str,
        get_method_name: str,
        *,
        unit: Any | None = None,
    ) -> float | None:
        has_method = getattr(action, has_method_name, None)
        if callable(has_method) and not has_method():
            return None

        getter = getattr(action, get_method_name)
        scalar = getter()
        try:
            return float(scalar.get_value(unit) if unit is not None else scalar.get_value())
        except Exception:
            return None
