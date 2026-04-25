"""Server-side adapter around the locally built Pulse physiology engine."""

from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from pulse_physiology_env.patient_state import (
    ArterialBloodGasResult,
    BasicMetabolicPanelResult,
    CompleteBloodCountResult,
    MentalStatus,
    PatientState,
    ScenarioDifficulty,
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
    from pulse.cdm.bag_valve_mask_actions import SEBagValveMaskAutomated, SEBagValveMaskConfiguration
    from pulse.cdm.engine import SEAction, SEDataRequest, SEDataRequestManager, eGate, eSide, eSwitch
    from pulse.cdm.mechanical_ventilator_actions import (
        SEMechanicalVentilatorContinuousPositiveAirwayPressure,
        SEMechanicalVentilatorPressureControl,
    )
    from pulse.cdm.patient_actions import (
        SEHemorrhage,
        SEIntubation,
        SENeedleDecompression,
        SEPericardialEffusion,
        SESubstanceCompoundInfusion,
        SESubstanceInfusion,
        SESupplementalOxygen,
        SETensionPneumothorax,
        eDevice,
        eHemorrhage_Compartment,
        eHemorrhage_Type,
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
        TimeUnit,
        VolumePerPressureUnit,
        VolumePerTimeUnit,
        VolumeUnit,
    )
except Exception as exc:  # pragma: no cover - exercised only on misconfigured hosts
    _PULSE_IMPORT_ERROR = exc


def _snake_case(value: str) -> str:
    compact = value.split(".")[-1].strip()
    first_pass = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", compact)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass).replace("__", "_").lower()


@dataclass
class _RuntimeContext:
    state_file: Path | None = None
    scenario_id: str = "baseline"
    scenario_difficulty: ScenarioDifficulty = "medium"
    patient_id: str = "standard_male"
    action_budget_remaining: int | None = None
    pending_diagnostics: dict[str, int] = field(default_factory=dict)
    ready_diagnostics: set[str] = field(default_factory=set)
    active_infusions: dict[str, float] = field(default_factory=dict)
    active_hemorrhages: dict[str, float] = field(default_factory=dict)
    active_tension_pneumothorax_sides: set[str] = field(default_factory=set)
    pericardial_effusion_rate_ml_per_min: float | None = None
    pain_sources: dict[str, float] = field(default_factory=dict)
    intubation_type: str = "Off"
    patient_position: str = "supine"
    oxygen_device: str | None = None
    oxygen_flow_lpm: float | None = None
    airway_support_mode: str | None = None
    fio2: float | None = None
    peep_cmh2o: float | None = None
    baseline_blood_volume_ml: float | None = None


class PulseEngineAdapter:
    """Loads Pulse states, advances simulation time, and synthesizes PatientState."""

    DEFAULT_STATE_FILENAME = "StandardMale@0s.json"
    DEFAULT_OXYGEN_SUPPLY_L = 1000.0
    DEFAULT_PATIENT_POSITION = "supine"
    DEFAULT_BVM_RESPIRATION_RATE_BPM = 12.0
    DEFAULT_BVM_IE_RATIO = 0.5
    DEFAULT_BVM_SQUEEZE_PRESSURE_CMH2O = 12.0
    DEFAULT_BVM_FIO2 = 0.6
    DEFAULT_CPAP_FIO2 = 0.6
    DEFAULT_CPAP_PEEP_CMH2O = 5.0
    DEFAULT_CPAP_PRESSURE_SUPPORT_CMH2O = 5.0
    DEFAULT_PC_FIO2 = 0.7
    DEFAULT_PC_PEEP_CMH2O = 5.0
    DEFAULT_PC_RESPIRATION_RATE_BPM = 14.0
    DEFAULT_PC_INSPIRATORY_PRESSURE_CMH2O = 16.0
    DEFAULT_PC_INSPIRATORY_PERIOD_S = 1.0
    DEFAULT_PRESSOR_MONITOR_SECONDS = 180.0
    DEFAULT_PERICARDIOCENTESIS_DRAIN_RATE_ML_PER_MIN = 150.0
    DIAGNOSTIC_DELAYS_S = {
        "get_blood_gas": 120,
        "get_cbc": 240,
        "get_bmp": 300,
    }

    _AIRWAY_SUPPORT_DISPLAY = {
        "bag_valve_mask": "bag_valve_mask",
        "cpap": "cpap",
        "pressure_control_ventilation": "pressure_control_ventilation",
    }
    _FLUID_COMPOUND_MAP = {
        "saline": "Saline",
        "blood": "Blood",
        "packed_rbc": "PackedRBC",
        "packed_rbcs": "PackedRBC",
        "prbc": "PackedRBC",
        "prbcs": "PackedRBC",
    }
    _PRESSOR_CONFIG = {
        "norepinephrine": {
            "substance": "Norepinephrine",
            "default_concentration_ug_per_ml": 1.0,
            "default_rate_ml_per_min": 7.8,
        },
        "phenylephrine": {
            "substance": "Phenylephrine",
            "default_concentration_ug_per_ml": 10.0,
            "default_rate_ml_per_min": 7.68,
        },
    }

    _SODIUM_MOLAR_MASS = 22.98976928
    _POTASSIUM_MOLAR_MASS = 39.0983
    _CALCIUM_MOLAR_MASS = 40.078

    _OXYGEN_DEVICE_MAP = {
        "nasal_cannula": eDevice.NasalCannula,
        "simple_mask": eDevice.SimpleMask,
        "non_rebreather_mask": eDevice.NonRebreatherMask,
    }
    _DEFAULT_OXYGEN_FLOW_LPM = {
        "nasal_cannula": 4.0,
        "simple_mask": 8.0,
        "non_rebreather_mask": 15.0,
    }
    _SIDE_MAP = {
        "left": eSide.Left,
        "right": eSide.Right,
    }
    _PNEUMOTHORAX_TYPE_MAP = {
        "open": eGate.Open,
        "closed": eGate.Closed,
    }
    _HEMORRHAGE_TYPE_MAP = {
        "external": eHemorrhage_Type.External,
        "internal": eHemorrhage_Type.Internal,
    }
    _INTUBATION_TYPE_MAP = {
        "off": eIntubationType.Off,
        "oropharyngeal": eIntubationType.Oropharyngeal,
        "oropharyngeal_airway": eIntubationType.Oropharyngeal,
        "nasopharyngeal": eIntubationType.Nasopharyngeal,
        "nasopharyngeal_airway": eIntubationType.Nasopharyngeal,
        "tracheal": eIntubationType.Tracheal,
        "endotracheal": eIntubationType.Tracheal,
        "endotracheal_tube": eIntubationType.Tracheal,
        "esophageal": eIntubationType.Esophageal,
        "left_mainstem": eIntubationType.LeftMainstem,
        "right_mainstem": eIntubationType.RightMainstem,
    }
    _AIRWAY_SUPPORT_NAMES = {
        eIntubationType.Off.name: None,
        eIntubationType.Oropharyngeal.name: "oropharyngeal_airway",
        eIntubationType.Nasopharyngeal.name: "nasopharyngeal_airway",
        eIntubationType.Tracheal.name: "endotracheal_tube",
        eIntubationType.Esophageal.name: "esophageal_intubation",
        eIntubationType.LeftMainstem.name: "left_mainstem_intubation",
        eIntubationType.RightMainstem.name: "right_mainstem_intubation",
    }
    _VALID_POSITIONS = {
        "supine",
        "upright",
        "trendelenburg",
        "recovery",
        "semi_fowler",
    }
    _HEMORRHAGE_COMPARTMENT_MAP = {
        "aorta": eHemorrhage_Compartment.Aorta,
        "brain": eHemorrhage_Compartment.Brain,
        "muscle": eHemorrhage_Compartment.Muscle,
        "large_intestine": eHemorrhage_Compartment.LargeIntestine,
        "left_arm": eHemorrhage_Compartment.LeftArm,
        "left_kidney": eHemorrhage_Compartment.LeftKidney,
        "left_leg": eHemorrhage_Compartment.LeftLeg,
        "liver": eHemorrhage_Compartment.Liver,
        "right_arm": eHemorrhage_Compartment.RightArm,
        "right_kidney": eHemorrhage_Compartment.RightKidney,
        "right_leg": eHemorrhage_Compartment.RightLeg,
        "skin": eHemorrhage_Compartment.Skin,
        "small_intestine": eHemorrhage_Compartment.SmallIntestine,
        "splanchnic": eHemorrhage_Compartment.Splanchnic,
        "spleen": eHemorrhage_Compartment.Spleen,
        "vena_cava": eHemorrhage_Compartment.VenaCava,
    }

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
        scenario_id: str = "baseline",
        scenario_difficulty: ScenarioDifficulty = "medium",
        patient_id: str | None = None,
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

        resolved_patient_id = patient_id or self._default_patient_id_from_state_file(resolved_state_file)
        self._runtime = _RuntimeContext(
            state_file=resolved_state_file,
            scenario_id=scenario_id,
            scenario_difficulty=scenario_difficulty,
            patient_id=resolved_patient_id,
            action_budget_remaining=(
                self._default_action_budget_remaining
                if action_budget_remaining is None
                else action_budget_remaining
            ),
            patient_position=self.DEFAULT_PATIENT_POSITION,
        )
        self._refresh_raw_metrics()
        self._runtime.baseline_blood_volume_ml = self._latest_raw_metrics.get("blood_volume_ml")
        self._last_lactate_mg_per_dl = self._latest_raw_metrics.get("lactate_mg_per_dl")

        if initial_actions:
            return self.apply_actions(initial_actions)

        return self.get_full_state()

    def set_scenario_context(
        self,
        *,
        scenario_id: str,
        scenario_difficulty: ScenarioDifficulty | None = None,
        patient_id: str | None = None,
    ) -> PatientState:
        """Update scenario metadata without reloading the engine."""

        self._runtime.scenario_id = scenario_id
        if scenario_difficulty is not None:
            self._runtime.scenario_difficulty = scenario_difficulty
        if patient_id is not None:
            self._runtime.patient_id = patient_id
        return self.get_full_state()

    def set_patient_position(self, position: str) -> PatientState:
        """Update the tracked patient position for downstream decision support."""

        canonical = position.strip().lower().replace("-", "_").replace(" ", "_")
        if canonical not in self._VALID_POSITIONS:
            valid = ", ".join(sorted(self._VALID_POSITIONS))
            raise ValueError(f"Unsupported patient position '{position}'. Expected one of: {valid}")

        self._runtime.patient_position = canonical
        return self.get_full_state()

    def apply_supplemental_oxygen(
        self,
        *,
        device: str,
        flow_lpm: float | None = None,
        supply_volume_l: float = DEFAULT_OXYGEN_SUPPLY_L,
        advance_time_seconds: float | None = 60.0,
    ) -> PatientState:
        """Apply supplemental oxygen using a Pulse native oxygen action."""

        device_key = device.strip().lower().replace("-", "_").replace(" ", "_")
        if device_key not in self._OXYGEN_DEVICE_MAP:
            valid = ", ".join(sorted(self._OXYGEN_DEVICE_MAP))
            raise ValueError(f"Unsupported oxygen device '{device}'. Expected one of: {valid}")

        resolved_flow_lpm = flow_lpm or self._DEFAULT_OXYGEN_FLOW_LPM[device_key]
        if resolved_flow_lpm <= 0:
            raise ValueError("flow_lpm must be greater than zero.")
        if supply_volume_l <= 0:
            raise ValueError("supply_volume_l must be greater than zero.")

        action = SESupplementalOxygen()
        action.set_device(self._OXYGEN_DEVICE_MAP[device_key])
        action.get_flow().set_value(resolved_flow_lpm, VolumePerTimeUnit.L_Per_min)
        action.get_volume().set_value(supply_volume_l, VolumeUnit.L)
        return self.apply_actions([action], advance_time_seconds=advance_time_seconds)

    def resolve_fluid_compound(self, fluid_type: str) -> str:
        fluid_key = fluid_type.strip().lower().replace("-", "_").replace(" ", "_")
        if fluid_key not in self._FLUID_COMPOUND_MAP:
            valid = ", ".join(sorted(self._FLUID_COMPOUND_MAP))
            raise ValueError(f"Unsupported fluid_type '{fluid_type}'. Expected one of: {valid}")
        return self._FLUID_COMPOUND_MAP[fluid_key]

    def infuse_compound(
        self,
        *,
        compound: str,
        bag_volume_ml: float,
        rate_ml_per_min: float,
        advance_time_seconds: float | None = None,
    ) -> PatientState:
        """Administer a Pulse compound infusion such as saline or blood."""

        if not compound or not compound.strip():
            raise ValueError("compound must be a non-empty string.")
        if bag_volume_ml <= 0:
            raise ValueError("bag_volume_ml must be greater than zero.")
        if rate_ml_per_min <= 0:
            raise ValueError("rate_ml_per_min must be greater than zero.")

        action = SESubstanceCompoundInfusion()
        action.set_compound(compound.strip())
        action.get_bag_volume().set_value(bag_volume_ml, VolumeUnit.mL)
        action.get_rate().set_value(rate_ml_per_min, VolumePerTimeUnit.mL_Per_min)

        delivery_duration_s = (bag_volume_ml / rate_ml_per_min) * 60.0
        monitor_window_s = delivery_duration_s if advance_time_seconds is None else advance_time_seconds
        state = self.apply_actions([action], advance_time_seconds=monitor_window_s)

        if monitor_window_s >= delivery_duration_s:
            self._runtime.active_infusions.pop(_snake_case(compound), None)
            state = self.get_full_state()
        return state

    def set_pressor(
        self,
        *,
        pressor: str,
        concentration_ug_per_ml: float | None = None,
        rate_ml_per_min: float | None = None,
        advance_time_seconds: float | None = DEFAULT_PRESSOR_MONITOR_SECONDS,
    ) -> PatientState:
        """Start, titrate, or stop a vasoactive infusion using Pulse's native drug action."""

        pressor_key = pressor.strip().lower().replace("-", "_").replace(" ", "_")
        if pressor_key not in self._PRESSOR_CONFIG:
            valid = ", ".join(sorted(self._PRESSOR_CONFIG))
            raise ValueError(f"Unsupported pressor '{pressor}'. Expected one of: {valid}")

        config = self._PRESSOR_CONFIG[pressor_key]
        resolved_concentration = (
            config["default_concentration_ug_per_ml"]
            if concentration_ug_per_ml is None
            else concentration_ug_per_ml
        )
        resolved_rate = config["default_rate_ml_per_min"] if rate_ml_per_min is None else rate_ml_per_min
        if resolved_concentration <= 0:
            raise ValueError("concentration_ug_per_ml must be greater than zero.")
        if resolved_rate < 0:
            raise ValueError("rate_ml_per_min cannot be negative.")

        action = SESubstanceInfusion()
        action.set_substance(config["substance"])
        action.get_concentration().set_value(resolved_concentration, MassPerVolumeUnit.ug_Per_mL)
        action.get_rate().set_value(resolved_rate, VolumePerTimeUnit.mL_Per_min)
        return self.apply_actions([action], advance_time_seconds=advance_time_seconds)

    def set_pericardial_effusion(
        self,
        *,
        effusion_rate_ml_per_min: float,
        advance_time_seconds: float | None = 60.0,
    ) -> PatientState:
        """Create, stop, or drain pericardial fluid using Pulse's native effusion action."""

        action = SEPericardialEffusion()
        action.get_effusion_rate().set_value(effusion_rate_ml_per_min, VolumePerTimeUnit.mL_Per_min)
        return self.apply_actions([action], advance_time_seconds=advance_time_seconds)

    def perform_pericardiocentesis(
        self,
        *,
        drain_rate_ml_per_min: float = DEFAULT_PERICARDIOCENTESIS_DRAIN_RATE_ML_PER_MIN,
        advance_time_seconds: float | None = 180.0,
    ) -> PatientState:
        """Drain pericardial fluid using a negative effusion rate."""

        if drain_rate_ml_per_min <= 0:
            raise ValueError("drain_rate_ml_per_min must be greater than zero.")

        return self.set_pericardial_effusion(
            effusion_rate_ml_per_min=-drain_rate_ml_per_min,
            advance_time_seconds=advance_time_seconds,
        )

    def apply_bag_valve_mask(
        self,
        *,
        fio2: float | None = None,
        peep_cmh2o: float | None = None,
        respiration_rate_bpm: float | None = None,
        inspiratory_expiratory_ratio: float | None = None,
        squeeze_pressure_cmh2o: float | None = None,
        squeeze_volume_ml: float | None = None,
        airway_adjunct: str | None = None,
        advance_time_seconds: float | None = 60.0,
    ) -> PatientState:
        """Provide bag-valve-mask support following Pulse's native equipment workflow."""

        resolved_fio2 = self.DEFAULT_BVM_FIO2 if fio2 is None else fio2
        resolved_rate = self.DEFAULT_BVM_RESPIRATION_RATE_BPM if respiration_rate_bpm is None else respiration_rate_bpm
        resolved_ie_ratio = (
            self.DEFAULT_BVM_IE_RATIO
            if inspiratory_expiratory_ratio is None
            else inspiratory_expiratory_ratio
        )
        resolved_pressure = (
            self.DEFAULT_BVM_SQUEEZE_PRESSURE_CMH2O
            if squeeze_pressure_cmh2o is None
            else squeeze_pressure_cmh2o
        )
        if not 0.21 <= resolved_fio2 <= 1.0:
            raise ValueError("fio2 must be between 0.21 and 1.0.")
        if resolved_rate <= 0:
            raise ValueError("respiration_rate_bpm must be greater than zero.")
        if resolved_ie_ratio <= 0:
            raise ValueError("inspiratory_expiratory_ratio must be greater than zero.")
        if resolved_pressure <= 0 and squeeze_volume_ml is None:
            raise ValueError("Provide a positive squeeze pressure or squeeze volume.")
        if squeeze_volume_ml is not None and squeeze_volume_ml <= 0:
            raise ValueError("squeeze_volume_ml must be greater than zero when provided.")
        if peep_cmh2o is not None and peep_cmh2o < 0:
            raise ValueError("peep_cmh2o cannot be negative.")

        adjunct = airway_adjunct
        if adjunct is None and not self._is_intubated():
            adjunct = "oropharyngeal"

        actions: list[SEAction] = []
        if adjunct:
            actions.append(self._build_intubation_action(adjunct))

        configuration = SEBagValveMaskConfiguration()
        bag = configuration.get_configuration()
        bag.set_connection(eSwitch.On)
        bag.get_fraction_inspired_gas("Oxygen").get_fraction_amount().set_value(resolved_fio2)
        if peep_cmh2o is not None:
            bag.get_valve_positive_end_expired_pressure().set_value(peep_cmh2o, PressureUnit.cmH2O)
        actions.append(configuration)

        automated = SEBagValveMaskAutomated()
        automated.get_breath_frequency().set_value(resolved_rate, FrequencyUnit.Per_min)
        automated.get_inspiratory_expiratory_ratio().set_value(resolved_ie_ratio)
        if squeeze_volume_ml is not None:
            automated.get_squeeze_volume().set_value(squeeze_volume_ml, VolumeUnit.mL)
        else:
            automated.get_squeeze_pressure().set_value(resolved_pressure, PressureUnit.cmH2O)
        actions.append(automated)
        return self.apply_actions(actions, advance_time_seconds=advance_time_seconds)

    def apply_cpap(
        self,
        *,
        fio2: float | None = None,
        peep_cmh2o: float | None = None,
        pressure_support_cmh2o: float | None = None,
        advance_time_seconds: float | None = 120.0,
    ) -> PatientState:
        """Provide non-invasive CPAP support through Pulse's mechanical ventilator mode."""

        resolved_fio2 = self.DEFAULT_CPAP_FIO2 if fio2 is None else fio2
        resolved_peep = self.DEFAULT_CPAP_PEEP_CMH2O if peep_cmh2o is None else peep_cmh2o
        resolved_support = (
            self.DEFAULT_CPAP_PRESSURE_SUPPORT_CMH2O
            if pressure_support_cmh2o is None
            else pressure_support_cmh2o
        )
        if not 0.21 <= resolved_fio2 <= 1.0:
            raise ValueError("fio2 must be between 0.21 and 1.0.")
        if resolved_peep < 0:
            raise ValueError("peep_cmh2o cannot be negative.")
        if resolved_support < 0:
            raise ValueError("pressure_support_cmh2o cannot be negative.")

        action = SEMechanicalVentilatorContinuousPositiveAirwayPressure()
        action.set_connection(eSwitch.On)
        action.get_fraction_inspired_oxygen().set_value(resolved_fio2)
        action.get_positive_end_expired_pressure().set_value(resolved_peep, PressureUnit.cmH2O)
        action.get_delta_pressure_support().set_value(resolved_support, PressureUnit.cmH2O)
        return self.apply_actions([action], advance_time_seconds=advance_time_seconds)

    def apply_pressure_control_ventilation(
        self,
        *,
        fio2: float | None = None,
        peep_cmh2o: float | None = None,
        inspiratory_pressure_cmh2o: float | None = None,
        respiration_rate_bpm: float | None = None,
        inspiratory_period_s: float | None = None,
        advance_time_seconds: float | None = 120.0,
    ) -> PatientState:
        """Provide invasive pressure-control ventilation using Pulse's native ventilator action."""

        resolved_fio2 = self.DEFAULT_PC_FIO2 if fio2 is None else fio2
        resolved_peep = self.DEFAULT_PC_PEEP_CMH2O if peep_cmh2o is None else peep_cmh2o
        resolved_pressure = (
            self.DEFAULT_PC_INSPIRATORY_PRESSURE_CMH2O
            if inspiratory_pressure_cmh2o is None
            else inspiratory_pressure_cmh2o
        )
        resolved_rate = (
            self.DEFAULT_PC_RESPIRATION_RATE_BPM
            if respiration_rate_bpm is None
            else respiration_rate_bpm
        )
        resolved_period = (
            self.DEFAULT_PC_INSPIRATORY_PERIOD_S
            if inspiratory_period_s is None
            else inspiratory_period_s
        )
        if not 0.21 <= resolved_fio2 <= 1.0:
            raise ValueError("fio2 must be between 0.21 and 1.0.")
        if resolved_peep < 0:
            raise ValueError("peep_cmh2o cannot be negative.")
        if resolved_pressure <= 0:
            raise ValueError("inspiratory_pressure_cmh2o must be greater than zero.")
        if resolved_rate <= 0:
            raise ValueError("respiration_rate_bpm must be greater than zero.")
        if resolved_period <= 0:
            raise ValueError("inspiratory_period_s must be greater than zero.")

        actions: list[SEAction] = [self._build_intubation_action("tracheal")]
        ventilator = SEMechanicalVentilatorPressureControl()
        ventilator.set_connection(eSwitch.On)
        ventilator.get_fraction_inspired_oxygen().set_value(resolved_fio2)
        ventilator.get_positive_end_expired_pressure().set_value(resolved_peep, PressureUnit.cmH2O)
        ventilator.get_inspiratory_pressure().set_value(resolved_pressure, PressureUnit.cmH2O)
        ventilator.get_respiration_rate().set_value(resolved_rate, FrequencyUnit.Per_min)
        ventilator.get_inspiratory_period().set_value(resolved_period, TimeUnit.s)
        actions.append(ventilator)
        return self.apply_actions(actions, advance_time_seconds=advance_time_seconds)

    def order_diagnostic(self, tool_name: str, *, delay_seconds: int | None = None) -> PatientState:
        """Schedule a delayed diagnostic and return the updated state."""

        if tool_name not in self.DIAGNOSTIC_DELAYS_S:
            valid = ", ".join(sorted(self.DIAGNOSTIC_DELAYS_S))
            raise ValueError(f"Unsupported diagnostic '{tool_name}'. Expected one of: {valid}")

        self._runtime.ready_diagnostics.discard(tool_name)
        self._runtime.pending_diagnostics[tool_name] = int(
            delay_seconds if delay_seconds is not None else self.DIAGNOSTIC_DELAYS_S[tool_name]
        )
        return self.get_full_state()

    def set_hemorrhage(
        self,
        compartment: str,
        *,
        flow_rate_ml_per_min: float | None = None,
        severity: float | None = None,
        hemorrhage_type: str = "external",
        advance_time_seconds: float | None = None,
    ) -> PatientState:
        """Start, modify, or stop a hemorrhage in a specific compartment."""

        compartment_key = compartment.strip().lower().replace("-", "_").replace(" ", "_")
        if compartment_key not in self._HEMORRHAGE_COMPARTMENT_MAP:
            valid = ", ".join(sorted(self._HEMORRHAGE_COMPARTMENT_MAP))
            raise ValueError(f"Unsupported hemorrhage compartment '{compartment}'. Expected one of: {valid}")

        hemorrhage_type_key = hemorrhage_type.strip().lower()
        if hemorrhage_type_key not in self._HEMORRHAGE_TYPE_MAP:
            valid = ", ".join(sorted(self._HEMORRHAGE_TYPE_MAP))
            raise ValueError(f"Unsupported hemorrhage type '{hemorrhage_type}'. Expected one of: {valid}")

        if flow_rate_ml_per_min is None and severity is None:
            flow_rate_ml_per_min = 0.0
        if flow_rate_ml_per_min is not None and flow_rate_ml_per_min < 0:
            raise ValueError("flow_rate_ml_per_min cannot be negative.")
        if severity is not None and not 0.0 <= severity <= 1.0:
            raise ValueError("severity must be between 0.0 and 1.0.")

        action = SEHemorrhage()
        action.set_type(self._HEMORRHAGE_TYPE_MAP[hemorrhage_type_key])
        action.set_compartment(self._HEMORRHAGE_COMPARTMENT_MAP[compartment_key])
        if flow_rate_ml_per_min is not None:
            action.get_flow_rate().set_value(flow_rate_ml_per_min, VolumePerTimeUnit.mL_Per_min)
        if severity is not None:
            action.get_severity().set_value(severity)
        return self.apply_actions([action], advance_time_seconds=advance_time_seconds)

    def set_tension_pneumothorax(
        self,
        side: str,
        *,
        severity: float,
        pneumothorax_type: str = "closed",
        advance_time_seconds: float | None = None,
    ) -> PatientState:
        """Create a tension pneumothorax using Pulse's native action."""

        side_key = side.strip().lower()
        if side_key not in self._SIDE_MAP:
            raise ValueError("side must be 'left' or 'right'.")
        if not 0.0 <= severity <= 1.0:
            raise ValueError("severity must be between 0.0 and 1.0.")
        type_key = pneumothorax_type.strip().lower()
        if type_key not in self._PNEUMOTHORAX_TYPE_MAP:
            valid = ", ".join(sorted(self._PNEUMOTHORAX_TYPE_MAP))
            raise ValueError(f"pneumothorax_type must be one of: {valid}")

        action = SETensionPneumothorax()
        action.set_side(self._SIDE_MAP[side_key])
        action.set_type(self._PNEUMOTHORAX_TYPE_MAP[type_key])
        action.get_severity().set_value(severity)
        return self.apply_actions([action], advance_time_seconds=advance_time_seconds)

    def apply_needle_decompression(
        self,
        side: str,
        *,
        advance_time_seconds: float | None = 90.0,
    ) -> PatientState:
        """Perform needle decompression on the specified hemithorax."""

        side_key = side.strip().lower()
        if side_key not in self._SIDE_MAP:
            raise ValueError("side must be 'left' or 'right'.")

        action = SENeedleDecompression()
        action.set_side(self._SIDE_MAP[side_key])
        action.set_state(eSwitch.On)
        return self.apply_actions([action], advance_time_seconds=advance_time_seconds)

    def set_intubation(
        self,
        kind: str,
        *,
        advance_time_seconds: float | None = 60.0,
    ) -> PatientState:
        """Apply or remove airway support using Pulse's intubation action."""

        kind_key = kind.strip().lower().replace("-", "_").replace(" ", "_")
        if kind_key not in self._INTUBATION_TYPE_MAP:
            valid = ", ".join(sorted(self._INTUBATION_TYPE_MAP))
            raise ValueError(f"Unsupported airway support '{kind}'. Expected one of: {valid}")

        action = self._build_intubation_action(kind_key)
        return self.apply_actions([action], advance_time_seconds=advance_time_seconds)

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
        if seconds == 0:
            return self.get_full_state()

        engine = self._require_engine()
        if not engine.advance_time_s(seconds):
            raise RuntimeError(f"Pulse failed to advance by {seconds} simulated seconds.")

        self._refresh_raw_metrics()
        self._tick_pending_diagnostics(seconds)
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

        arterial_pco2 = metrics.get("arterial_carbon_dioxide_pressure_mmhg")
        blood_ph = metrics.get("blood_ph")
        bicarbonate = self._derive_bicarbonate_meq_per_l(blood_ph, arterial_pco2)
        base_excess = metrics.get("base_excess_meq_per_l")
        base_deficit = max(-base_excess, 0.0) if base_excess is not None else None

        abg_result = ArterialBloodGasResult(
            ph=blood_ph,
            partial_pressure_of_oxygen_mmhg=metrics.get("arterial_oxygen_pressure_mmhg"),
            partial_pressure_of_carbon_dioxide_mmhg=arterial_pco2,
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
        mental_status = self._derive_mental_status(metrics, is_alive)
        shock_index = self._derive_shock_index(
            metrics.get("heart_rate_bpm"),
            metrics.get("systolic_bp_mmhg"),
        )
        state = PatientState(
            scenario_id=self._runtime.scenario_id,
            scenario_difficulty=self._runtime.scenario_difficulty,
            patient_id=self._runtime.patient_id,
            sim_time_s=metrics.get("simulated_time_seconds") or 0.0,
            heart_rate_bpm=metrics.get("heart_rate_bpm"),
            systolic_bp_mmhg=metrics.get("systolic_bp_mmhg"),
            diastolic_bp_mmhg=metrics.get("diastolic_bp_mmhg"),
            mean_arterial_pressure_mmhg=metrics.get("mean_arterial_pressure_mmhg"),
            cardiac_output_l_per_min=metrics.get("cardiac_output_l_per_min"),
            spo2=metrics.get("oxygen_saturation"),
            respiration_rate_bpm=metrics.get("respiration_rate_bpm"),
            blood_volume_ml=blood_volume_ml,
            mental_status=mental_status,
            active_alerts=self._derive_alerts(metrics, is_alive, shock_index),
            done=not is_alive,
            etco2_mmhg=metrics.get("end_tidal_carbon_dioxide_pressure_mmhg"),
            tidal_volume_ml=metrics.get("tidal_volume_ml"),
            breath_sounds=self._derive_breath_sounds(),
            core_temperature_c=metrics.get("core_temperature_c"),
            shock_index=shock_index,
            lactate_trend=self._derive_lactate_trend(metrics.get("lactate_mg_per_dl")),
            position=self._runtime.patient_position,
            oxygen_device=self._runtime.oxygen_device,
            oxygen_flow_lpm=self._runtime.oxygen_flow_lpm,
            airway_support=self._derive_airway_support_name(),
            intubated=self._is_intubated(),
            abg_result=abg_result,
            cbc_result=cbc_result,
            bmp_result=bmp_result,
            pending_diagnostics=dict(self._runtime.pending_diagnostics),
            ready_diagnostics=sorted(self._runtime.ready_diagnostics),
            active_infusions=dict(self._runtime.active_infusions),
            active_hemorrhages=dict(self._runtime.active_hemorrhages),
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
            (
                "systolic_bp_mmhg",
                SEDataRequest.create_physiology_request("SystolicArterialPressure", unit=PressureUnit.mmHg),
            ),
            (
                "diastolic_bp_mmhg",
                SEDataRequest.create_physiology_request("DiastolicArterialPressure", unit=PressureUnit.mmHg),
            ),
            (
                "mean_arterial_pressure_mmhg",
                SEDataRequest.create_physiology_request("MeanArterialPressure", unit=PressureUnit.mmHg),
            ),
            (
                "cardiac_output_l_per_min",
                SEDataRequest.create_physiology_request("CardiacOutput", unit=VolumePerTimeUnit.L_Per_min),
            ),
            ("stroke_volume_ml", SEDataRequest.create_physiology_request("HeartStrokeVolume", unit=VolumeUnit.mL)),
            (
                "systemic_vascular_resistance_mmhg_min_per_l",
                SEDataRequest.create_physiology_request(
                    "SystemicVascularResistance",
                    unit=PressureTimePerVolumeUnit.mmHg_min_Per_L,
                ),
            ),
            ("oxygen_saturation", SEDataRequest.create_physiology_request("OxygenSaturation")),
            (
                "respiration_rate_bpm",
                SEDataRequest.create_physiology_request("RespirationRate", unit=FrequencyUnit.Per_min),
            ),
            (
                "end_tidal_carbon_dioxide_pressure_mmhg",
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
            (
                "base_excess_meq_per_l",
                SEDataRequest.create_physiology_request("BaseExcess", unit=AmountPerVolumeUnit.mEq_Per_L),
            ),
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
                "arterial_oxygen_pressure_mmhg",
                SEDataRequest.create_physiology_request("ArterialOxygenPressure", unit=PressureUnit.mmHg),
            ),
            (
                "arterial_carbon_dioxide_pressure_mmhg",
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
            self._runtime.airway_support_mode = None
            return

        if isinstance(action, SESupplementalOxygen):
            device_name = action.get_device().name if hasattr(action.get_device(), "name") else str(action.get_device())
            self._runtime.oxygen_device = _snake_case(device_name) if device_name and "Null" not in device_name else None
            flow_lpm = self._get_scalar_value(action, "has_flow", "get_flow", unit=VolumePerTimeUnit.L_Per_min)
            self._runtime.oxygen_flow_lpm = flow_lpm
            return

        if isinstance(action, SEBagValveMaskConfiguration):
            configuration = action.get_configuration()
            self._runtime.airway_support_mode = "bag_valve_mask"
            self._runtime.fio2 = self._extract_substance_fraction(configuration, "Oxygen")
            self._runtime.peep_cmh2o = self._get_scalar_value(
                configuration,
                "has_valve_positive_end_expired_pressure",
                "get_valve_positive_end_expired_pressure",
                unit=PressureUnit.cmH2O,
            )
            return

        if isinstance(action, SEBagValveMaskAutomated):
            self._runtime.airway_support_mode = "bag_valve_mask"
            return

        if isinstance(action, SEMechanicalVentilatorContinuousPositiveAirwayPressure):
            if action.get_connection() == eSwitch.On:
                self._runtime.airway_support_mode = "cpap"
                self._runtime.fio2 = self._get_scalar_value(
                    action,
                    "has_fraction_inspired_oxygen",
                    "get_fraction_inspired_oxygen",
                )
                self._runtime.peep_cmh2o = self._get_scalar_value(
                    action,
                    "has_positive_end_expired_pressure",
                    "get_positive_end_expired_pressure",
                    unit=PressureUnit.cmH2O,
                )
            return

        if isinstance(action, SEMechanicalVentilatorPressureControl):
            if action.get_connection() == eSwitch.On:
                self._runtime.airway_support_mode = "pressure_control_ventilation"
                self._runtime.fio2 = self._get_scalar_value(
                    action,
                    "has_fraction_inspired_oxygen",
                    "get_fraction_inspired_oxygen",
                )
                self._runtime.peep_cmh2o = self._get_scalar_value(
                    action,
                    "has_positive_end_expired_pressure",
                    "get_positive_end_expired_pressure",
                    unit=PressureUnit.cmH2O,
                )
            return

        if isinstance(action, SENeedleDecompression) and action.has_side():
            side = _snake_case(action.get_side().name)
            self._runtime.active_tension_pneumothorax_sides.discard(side)
            self._runtime.pain_sources.pop(f"tension_pneumothorax:{side}", None)
            return

        if isinstance(action, SETensionPneumothorax) and action.has_side():
            side = _snake_case(action.get_side().name)
            severity = self._get_scalar_value(action, "has_severity", "get_severity")
            self._runtime.active_tension_pneumothorax_sides.add(side)
            self._runtime.pain_sources[f"tension_pneumothorax:{side}"] = severity or 0.8
            return

        if isinstance(action, SEPericardialEffusion):
            rate_ml_per_min = self._get_scalar_value(
                action,
                "has_effusion_rate",
                "get_effusion_rate",
                unit=VolumePerTimeUnit.mL_Per_min,
            )
            if rate_ml_per_min is None or abs(rate_ml_per_min) < 1e-6:
                self._runtime.pericardial_effusion_rate_ml_per_min = None
                self._runtime.pain_sources.pop("pericardial_effusion", None)
            else:
                self._runtime.pericardial_effusion_rate_ml_per_min = rate_ml_per_min
                self._runtime.pain_sources["pericardial_effusion"] = min(abs(rate_ml_per_min) / 150.0, 1.0)
            return

        if isinstance(action, SEHemorrhage) and action.has_compartment():
            compartment = self._normalize_compartment_name(action.get_compartment())
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

        if isinstance(action, SESubstanceCompoundInfusion):
            compound = action.get_compound()
            if compound:
                rate_ml_per_min = self._get_scalar_value(
                    action,
                    "has_rate",
                    "get_rate",
                    unit=VolumePerTimeUnit.mL_Per_min,
                )
                name = _snake_case(str(compound))
                if rate_ml_per_min is not None and rate_ml_per_min > 0:
                    self._runtime.active_infusions[name] = rate_ml_per_min
                else:
                    self._runtime.active_infusions.pop(name, None)
            return

        if isinstance(action, SESubstanceInfusion) and action.has_substance():
            substance = str(action.get_substance())
            rate_ml_per_min = self._get_scalar_value(
                action,
                "has_rate",
                "get_rate",
                unit=VolumePerTimeUnit.mL_Per_min,
            )
            name = _snake_case(substance)
            if rate_ml_per_min is not None and rate_ml_per_min > 0:
                self._runtime.active_infusions[name] = rate_ml_per_min
            else:
                self._runtime.active_infusions.pop(name, None)

    def _is_alive_from_metrics(self, metrics: dict[str, float | None]) -> bool:
        if self._has_terminal_event():
            return False

        heart_rate = metrics.get("heart_rate_bpm")
        map_mmhg = metrics.get("mean_arterial_pressure_mmhg")
        spo2 = metrics.get("oxygen_saturation")
        etco2 = metrics.get("end_tidal_carbon_dioxide_pressure_mmhg")

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
        elif self._runtime.intubation_type == eIntubationType.Esophageal.name:
            left_absent = True
            right_absent = True

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

    def _derive_airway_support_name(self) -> str | None:
        if self._runtime.airway_support_mode:
            return self._AIRWAY_SUPPORT_DISPLAY.get(
                self._runtime.airway_support_mode,
                self._runtime.airway_support_mode,
            )
        return self._AIRWAY_SUPPORT_NAMES.get(self._runtime.intubation_type)

    def _is_intubated(self) -> bool:
        return self._runtime.intubation_type in {
            eIntubationType.Esophageal.name,
            eIntubationType.LeftMainstem.name,
            eIntubationType.RightMainstem.name,
            eIntubationType.Tracheal.name,
        }

    def _tick_pending_diagnostics(self, elapsed_seconds: float) -> None:
        if not self._runtime.pending_diagnostics:
            return

        completed: list[str] = []
        for tool_name, remaining in list(self._runtime.pending_diagnostics.items()):
            updated_remaining = max(0, int(math.ceil(remaining - elapsed_seconds)))
            if updated_remaining == 0:
                completed.append(tool_name)
                self._runtime.pending_diagnostics.pop(tool_name, None)
            else:
                self._runtime.pending_diagnostics[tool_name] = updated_remaining

        for tool_name in completed:
            self._runtime.ready_diagnostics.add(tool_name)

    def _derive_shock_index(self, heart_rate_bpm: float | None, systolic_bp_mmhg: float | None) -> float | None:
        if heart_rate_bpm is None or systolic_bp_mmhg is None or systolic_bp_mmhg <= 0:
            return None
        return heart_rate_bpm / systolic_bp_mmhg

    def _derive_lactate_trend(self, current_lactate_mg_per_dl: float | None) -> str:
        previous = self._last_lactate_mg_per_dl
        if current_lactate_mg_per_dl is None or previous is None:
            return "stable"

        delta = current_lactate_mg_per_dl - previous
        if abs(delta) < 0.25:
            return "stable"
        return "improving" if delta < 0 else "worsening"

    def _derive_mental_status(
        self,
        metrics: dict[str, float | None],
        is_alive: bool,
    ) -> MentalStatus:
        if not is_alive:
            return "unresponsive"

        score = 1.0 - (metrics.get("sedation_level") or 0.0)
        map_mmhg = metrics.get("mean_arterial_pressure_mmhg")
        spo2 = metrics.get("oxygen_saturation")

        if map_mmhg is not None:
            if map_mmhg < 45:
                score = min(score, 0.1)
            elif map_mmhg < 55:
                score = min(score, 0.25)
            elif map_mmhg < 65:
                score = min(score, 0.55)
            elif map_mmhg < 75:
                score = min(score, 0.8)

        if spo2 is not None:
            if spo2 < 0.75:
                score = min(score, 0.1)
            elif spo2 < 0.85:
                score = min(score, 0.25)
            elif spo2 < 0.9:
                score = min(score, 0.55)
            elif spo2 < 0.94:
                score = min(score, 0.8)

        if score >= 0.85:
            return "alert"
        if score >= 0.6:
            return "verbal"
        if score >= 0.3:
            return "pain"
        return "unresponsive"

    def _derive_alerts(
        self,
        metrics: dict[str, float | None],
        is_alive: bool,
        shock_index: float | None,
    ) -> list[str]:
        if not is_alive:
            return ["cardiac_arrest"]

        alerts: list[str] = []
        heart_rate = metrics.get("heart_rate_bpm")
        systolic = metrics.get("systolic_bp_mmhg")
        map_mmhg = metrics.get("mean_arterial_pressure_mmhg")
        spo2 = metrics.get("oxygen_saturation")
        respiration_rate = metrics.get("respiration_rate_bpm")
        blood_ph = metrics.get("blood_ph")

        if heart_rate is not None and heart_rate >= 100:
            alerts.append("tachycardia")
        if heart_rate is not None and heart_rate <= 50:
            alerts.append("bradycardia")
        if systolic is not None and systolic < 90 or map_mmhg is not None and map_mmhg < 65:
            alerts.append("hypotension")
        if spo2 is not None and spo2 < 0.92:
            alerts.append("hypoxemia")
        if spo2 is not None and spo2 < 0.85:
            alerts.append("severe_hypoxemia")
        if respiration_rate is not None and respiration_rate >= 24:
            alerts.append("tachypnea")
        if respiration_rate is not None and respiration_rate <= 8:
            alerts.append("bradypnea")
        if shock_index is not None and shock_index >= 0.9:
            alerts.append("shock_index_elevated")
        if self._runtime.active_hemorrhages:
            alerts.append("active_hemorrhage")
        if self._runtime.active_tension_pneumothorax_sides:
            alerts.append("possible_tension_pneumothorax")
        if self._runtime.intubation_type == eIntubationType.Esophageal.name:
            alerts.append("misplaced_airway")
        if self._runtime.pending_diagnostics:
            alerts.append("diagnostics_pending")
        effusion_rate = self._runtime.pericardial_effusion_rate_ml_per_min
        if effusion_rate is not None:
            if effusion_rate > 0:
                alerts.append("active_pericardial_effusion")
                if map_mmhg is not None and map_mmhg < 65:
                    alerts.append("possible_cardiac_tamponade")
            elif effusion_rate < 0:
                alerts.append("pericardiocentesis_in_progress")
        breath_sounds = self._derive_breath_sounds()
        if breath_sounds == "absent bilateral":
            alerts.append("bilateral_absent_breath_sounds")
        elif breath_sounds != "present bilateral":
            alerts.append("unilateral_absent_breath_sounds")
        if blood_ph is not None and blood_ph < 7.35:
            alerts.append("acidosis")

        return list(dict.fromkeys(alerts))

    @staticmethod
    def _derive_bicarbonate_meq_per_l(ph: float | None, arterial_pco2_mmhg: float | None) -> float | None:
        if ph is None or arterial_pco2_mmhg is None:
            return None
        return 0.03 * arterial_pco2_mmhg * (10 ** (ph - 6.1))

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

    def _build_intubation_action(self, kind: str) -> SEIntubation:
        kind_key = kind.strip().lower().replace("-", "_").replace(" ", "_")
        if kind_key not in self._INTUBATION_TYPE_MAP:
            valid = ", ".join(sorted(self._INTUBATION_TYPE_MAP))
            raise ValueError(f"Unsupported airway support '{kind}'. Expected one of: {valid}")

        action = SEIntubation()
        action.set_type(self._INTUBATION_TYPE_MAP[kind_key])
        return action

    @staticmethod
    def _extract_substance_fraction(configuration: Any, substance_name: str) -> float | None:
        try:
            fraction = configuration.get_fraction_inspired_gas(substance_name)
            amount = fraction.get_fraction_amount()
            if amount is None:
                return None
            value = amount.get_value()
            return None if value is None else float(value)
        except Exception:
            return None

    @staticmethod
    def _default_patient_id_from_state_file(state_file: Path) -> str:
        stem = state_file.stem.split("@")[0]
        return _snake_case(stem)

    @staticmethod
    def _normalize_compartment_name(compartment: Any) -> str:
        name = getattr(compartment, "name", str(compartment))
        return _snake_case(name)
