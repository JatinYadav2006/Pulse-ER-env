"""Deterministic mock scenarios for the Person 2 workstream."""

from dataclasses import dataclass

from ..patient_state import PatientState


@dataclass(frozen=True)
class MockScenarioDefinition:
    """Static data that drives deterministic mock physiology behavior."""

    scenario_id: str
    description: str
    initial_state: PatientState
    injury_severity: float
    deterioration_per_30s: dict[str, float]
    tool_effects: dict[str, dict[str, float]]
    recommended_actions: tuple[str, ...]


BASELINE_STABLE = MockScenarioDefinition(
    scenario_id="baseline_stable",
    description="Stable patient with near-normal vitals and low intervention urgency.",
    initial_state=PatientState(
        scenario_id="baseline_stable",
        patient_id="standard_male",
        sim_time_s=0.0,
        heart_rate_bpm=72.0,
        systolic_bp_mmhg=114.0,
        diastolic_bp_mmhg=74.0,
        spo2=0.974,
        respiration_rate_bpm=12.0,
        blood_volume_ml=5489.0,
        mental_status="alert",
        active_alerts=[],
        done=False,
    ),
    injury_severity=0.2,
    deterioration_per_30s={
        "heart_rate_bpm": 0.5,
        "systolic_bp_mmhg": -0.5,
        "diastolic_bp_mmhg": -0.3,
        "spo2": -0.002,
        "respiration_rate_bpm": 0.2,
    },
    tool_effects={
        "get_vitals": {},
        "advance_time": {},
        "summarize_state": {},
        "check_deterioration": {},
        "recommend_next_step": {},
    },
    recommended_actions=("get_vitals", "check_deterioration", "advance_time"),
)


RESPIRATORY_DISTRESS = MockScenarioDefinition(
    scenario_id="respiratory_distress",
    description="Hypoxemic patient with tachypnea who improves with airway and oxygen support.",
    initial_state=PatientState(
        scenario_id="respiratory_distress",
        patient_id="standard_male",
        sim_time_s=0.0,
        heart_rate_bpm=108.0,
        systolic_bp_mmhg=110.0,
        diastolic_bp_mmhg=70.0,
        spo2=0.89,
        respiration_rate_bpm=30.0,
        blood_volume_ml=5450.0,
        mental_status="alert",
        active_alerts=["hypoxemia", "tachypnea"],
        done=False,
    ),
    injury_severity=0.65,
    deterioration_per_30s={
        "heart_rate_bpm": 3.0,
        "systolic_bp_mmhg": -2.0,
        "diastolic_bp_mmhg": -1.0,
        "spo2": -0.03,
        "respiration_rate_bpm": 2.0,
    },
    tool_effects={
        "get_vitals": {},
        "advance_time": {},
        "give_oxygen": {"spo2": 0.05, "respiration_rate_bpm": -4.0, "heart_rate_bpm": -3.0},
        "position_patient": {"spo2": 0.02, "respiration_rate_bpm": -2.0},
        "airway_support": {"spo2": 0.04, "respiration_rate_bpm": -5.0, "heart_rate_bpm": -2.0},
        "summarize_state": {},
        "check_deterioration": {},
        "recommend_next_step": {},
    },
    recommended_actions=("get_vitals", "give_oxygen", "position_patient", "airway_support"),
)


HEMORRHAGIC_SHOCK = MockScenarioDefinition(
    scenario_id="hemorrhagic_shock",
    description="Bleeding patient with falling pressure and blood volume who responds to bleeding control and fluids.",
    initial_state=PatientState(
        scenario_id="hemorrhagic_shock",
        patient_id="standard_male",
        sim_time_s=0.0,
        heart_rate_bpm=124.0,
        systolic_bp_mmhg=92.0,
        diastolic_bp_mmhg=58.0,
        spo2=0.95,
        respiration_rate_bpm=26.0,
        blood_volume_ml=4700.0,
        mental_status="alert",
        active_alerts=["tachycardia", "hypotension", "blood_loss", "tachypnea"],
        done=False,
    ),
    injury_severity=0.85,
    deterioration_per_30s={
        "heart_rate_bpm": 4.0,
        "systolic_bp_mmhg": -5.0,
        "diastolic_bp_mmhg": -3.0,
        "spo2": -0.01,
        "respiration_rate_bpm": 1.5,
        "blood_volume_ml": -180.0,
    },
    tool_effects={
        "get_vitals": {},
        "advance_time": {},
        "give_fluids": {
            "systolic_bp_mmhg": 10.0,
            "diastolic_bp_mmhg": 5.0,
            "heart_rate_bpm": -5.0,
            "respiration_rate_bpm": -1.0,
            "blood_volume_ml": 350.0,
        },
        "control_bleeding": {
            "systolic_bp_mmhg": 6.0,
            "diastolic_bp_mmhg": 3.0,
            "heart_rate_bpm": -4.0,
            "blood_volume_ml": 180.0,
        },
        "position_patient": {
            "systolic_bp_mmhg": 3.0,
            "diastolic_bp_mmhg": 1.5,
            "respiration_rate_bpm": -1.5,
        },
        "give_oxygen": {"spo2": 0.03, "heart_rate_bpm": -1.0, "respiration_rate_bpm": -1.0},
        "summarize_state": {},
        "check_deterioration": {},
        "recommend_next_step": {},
    },
    recommended_actions=(
        "get_vitals",
        "control_bleeding",
        "give_fluids",
        "give_oxygen",
        "position_patient",
        "check_deterioration",
    ),
)


MOCK_SCENARIOS: dict[str, MockScenarioDefinition] = {
    BASELINE_STABLE.scenario_id: BASELINE_STABLE,
    RESPIRATORY_DISTRESS.scenario_id: RESPIRATORY_DISTRESS,
    HEMORRHAGIC_SHOCK.scenario_id: HEMORRHAGIC_SHOCK,
}

DEFAULT_MOCK_SCENARIO_ID = BASELINE_STABLE.scenario_id
