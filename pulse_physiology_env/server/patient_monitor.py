"""Backend monitor payload builder for the hackathon demo dashboard."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

from pulse_physiology_env.patient_state import PatientState

from .reward_engine import ActionRecord

MonitorStatus = Literal["stable", "warning", "critical", "improving", "info"]


@dataclass(frozen=True)
class MonitorTile:
    """One monitor tile for the eventual UI."""

    key: str
    label: str
    value: str
    status: MonitorStatus
    trend_delta: float | None = None


@dataclass(frozen=True)
class MonitorTrendPoint:
    """One sampled point for the patient monitor charts."""

    sim_time_s: float
    mean_arterial_pressure_mmhg: float | None
    spo2: float | None
    heart_rate_bpm: float | None
    respiration_rate_bpm: float | None
    etco2_mmhg: float | None


@dataclass(frozen=True)
class MonitorEvent:
    """Recent event for the intervention and milestone feed."""

    sim_time_s: float
    label: str
    detail: str
    tone: MonitorStatus


@dataclass(frozen=True)
class PatientMonitorPayload:
    """Structured monitor payload that a frontend can render directly."""

    headline: str
    status_banner: str
    scenario_2_moment: str | None
    tiles: list[MonitorTile]
    alerts: list[str]
    active_interventions: list[str]
    diagnostic_status: dict[str, Any]
    recent_events: list[MonitorEvent]
    trends: list[MonitorTrendPoint]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""

        return asdict(self)


class PatientMonitorVisualization:
    """Transforms environment state into a demo-friendly monitor payload."""

    MAX_TREND_POINTS = 24
    MAX_RECENT_EVENTS = 6

    def build(
        self,
        *,
        history: Sequence[PatientState],
        action_history: Sequence[ActionRecord],
        current_state: PatientState,
    ) -> PatientMonitorPayload:
        previous_state = history[-2] if len(history) >= 2 else None
        return PatientMonitorPayload(
            headline=self._build_headline(previous_state, current_state, action_history),
            status_banner=self._build_status_banner(current_state),
            scenario_2_moment=self._build_scenario_2_moment(previous_state, current_state, action_history),
            tiles=self._build_tiles(previous_state, current_state),
            alerts=list(current_state.active_alerts),
            active_interventions=self._build_active_interventions(current_state),
            diagnostic_status={
                "pending": dict(current_state.pending_diagnostics),
                "ready": list(current_state.ready_diagnostics),
            },
            recent_events=self._build_recent_events(action_history),
            trends=self._build_trends(history),
        )

    def _build_headline(
        self,
        previous_state: PatientState | None,
        current_state: PatientState,
        action_history: Sequence[ActionRecord],
    ) -> str:
        if previous_state is not None:
            if current_state.spo2 is not None and previous_state.spo2 is not None:
                spo2_jump = current_state.spo2 - previous_state.spo2
                if spo2_jump >= 0.08 and self._last_action_has_tag(action_history, "needle_decompression"):
                    return (
                        "Respiratory rescue visible: SpO2 improved from "
                        f"{previous_state.spo2:.2f} to {current_state.spo2:.2f} after decompression."
                    )
            if (
                current_state.mean_arterial_pressure_mmhg is not None
                and previous_state.mean_arterial_pressure_mmhg is not None
                and previous_state.mean_arterial_pressure_mmhg < 65.0 <= current_state.mean_arterial_pressure_mmhg
            ):
                return "Perfusion restored: MAP crossed the 65 mmHg resuscitation threshold."

        if current_state.spo2 is not None and current_state.spo2 < 0.88:
            return "Critical respiratory compromise on monitor; airway and chest interventions should be obvious."
        if current_state.mean_arterial_pressure_mmhg is not None and current_state.mean_arterial_pressure_mmhg < 55.0:
            return "Critical perfusion deficit on monitor; immediate resuscitation sequence is needed."
        return "Live trauma monitor ready for the decompression-first demo moment."

    @staticmethod
    def _build_status_banner(current_state: PatientState) -> str:
        if "cardiac_arrest" in current_state.active_alerts:
            return "Cardiac arrest physiology active."
        if current_state.spo2 is not None and current_state.spo2 < 0.88:
            return "Oxygenation is critical."
        if current_state.mean_arterial_pressure_mmhg is not None and current_state.mean_arterial_pressure_mmhg < 55.0:
            return "Perfusion is critical."
        if current_state.active_alerts:
            return "Active trauma alerts require attention."
        return "Physiology is currently stable enough for continued observation."

    def _build_scenario_2_moment(
        self,
        previous_state: PatientState | None,
        current_state: PatientState,
        action_history: Sequence[ActionRecord],
    ) -> str | None:
        if previous_state is None:
            return None
        if not self._last_action_has_tag(action_history, "needle_decompression"):
            return None
        if previous_state.spo2 is None or current_state.spo2 is None:
            return None
        improvement = current_state.spo2 - previous_state.spo2
        if improvement < 0.05:
            return None
        return (
            "Scenario 2 moment: decompression produced a visible SpO2 jump "
            f"from {previous_state.spo2:.2f} to {current_state.spo2:.2f}."
        )

    def _build_tiles(
        self,
        previous_state: PatientState | None,
        current_state: PatientState,
    ) -> list[MonitorTile]:
        return [
            self._tile(
                key="heart_rate_bpm",
                label="Heart Rate",
                value=self._format_value(current_state.heart_rate_bpm, "bpm"),
                status=self._status_for_heart_rate(current_state.heart_rate_bpm),
                trend_delta=self._delta(previous_state, current_state, "heart_rate_bpm"),
            ),
            self._tile(
                key="mean_arterial_pressure_mmhg",
                label="MAP",
                value=self._format_value(current_state.mean_arterial_pressure_mmhg, "mmHg"),
                status=self._status_for_map(current_state.mean_arterial_pressure_mmhg),
                trend_delta=self._delta(previous_state, current_state, "mean_arterial_pressure_mmhg"),
            ),
            self._tile(
                key="spo2",
                label="SpO2",
                value=self._format_fraction(current_state.spo2),
                status=self._status_for_spo2(current_state.spo2),
                trend_delta=self._delta(previous_state, current_state, "spo2"),
            ),
            self._tile(
                key="respiration_rate_bpm",
                label="Respiratory Rate",
                value=self._format_value(current_state.respiration_rate_bpm, "bpm"),
                status=self._status_for_respiration(current_state.respiration_rate_bpm),
                trend_delta=self._delta(previous_state, current_state, "respiration_rate_bpm"),
            ),
            self._tile(
                key="etco2_mmhg",
                label="EtCO2",
                value=self._format_value(current_state.etco2_mmhg, "mmHg"),
                status=self._status_for_etco2(current_state.etco2_mmhg),
                trend_delta=self._delta(previous_state, current_state, "etco2_mmhg"),
            ),
            self._tile(
                key="shock_index",
                label="Shock Index",
                value=self._format_number(current_state.shock_index, precision=2),
                status=self._status_for_shock_index(current_state.shock_index),
                trend_delta=self._delta(previous_state, current_state, "shock_index"),
            ),
        ]

    def _build_active_interventions(self, current_state: PatientState) -> list[str]:
        interventions: list[str] = []
        if current_state.oxygen_device is not None:
            flow = self._format_number(current_state.oxygen_flow_lpm)
            interventions.append(f"Oxygen: {current_state.oxygen_device} at {flow} L/min")
        if current_state.airway_support is not None:
            interventions.append(f"Airway support: {current_state.airway_support}")
        if current_state.intubated:
            interventions.append("Airway secured")
        for name, rate in sorted(current_state.active_infusions.items()):
            interventions.append(f"Infusion: {name} at {self._format_number(rate)}")
        for site, flow in sorted(current_state.active_hemorrhages.items()):
            interventions.append(f"Active hemorrhage: {site} at {self._format_number(flow)} mL/min")
        return interventions

    def _build_recent_events(self, action_history: Sequence[ActionRecord]) -> list[MonitorEvent]:
        events: list[MonitorEvent] = []
        for record in action_history[-self.MAX_RECENT_EVENTS :]:
            tone: MonitorStatus = "info"
            if "needle_decompression" in record.tags or "pericardiocentesis" in record.tags:
                tone = "improving"
            elif record.tool_name in {"perform_cpr", "induce_cardiac_arrest"}:
                tone = "critical"
            elif record.tool_name in {"apply_tourniquet", "control_bleeding", "apply_direct_pressure"}:
                tone = "warning"
            events.append(
                MonitorEvent(
                    sim_time_s=record.sim_time_s,
                    label=record.tool_name,
                    detail=f"successful={record.success}",
                    tone=tone,
                )
            )
        return events

    def _build_trends(self, history: Sequence[PatientState]) -> list[MonitorTrendPoint]:
        trend_states = history[-self.MAX_TREND_POINTS :]
        return [
            MonitorTrendPoint(
                sim_time_s=state.sim_time_s,
                mean_arterial_pressure_mmhg=state.mean_arterial_pressure_mmhg,
                spo2=state.spo2,
                heart_rate_bpm=state.heart_rate_bpm,
                respiration_rate_bpm=state.respiration_rate_bpm,
                etco2_mmhg=state.etco2_mmhg,
            )
            for state in trend_states
        ]

    @staticmethod
    def _tile(
        *,
        key: str,
        label: str,
        value: str,
        status: MonitorStatus,
        trend_delta: float | None,
    ) -> MonitorTile:
        return MonitorTile(
            key=key,
            label=label,
            value=value,
            status=status,
            trend_delta=None if trend_delta is None else round(trend_delta, 3),
        )

    @staticmethod
    def _delta(
        previous_state: PatientState | None,
        current_state: PatientState,
        field_name: str,
    ) -> float | None:
        if previous_state is None:
            return None
        previous_value = getattr(previous_state, field_name)
        current_value = getattr(current_state, field_name)
        if previous_value is None or current_value is None:
            return None
        return float(current_value) - float(previous_value)

    @staticmethod
    def _format_value(value: float | None, unit: str) -> str:
        if value is None:
            return "n/a"
        return f"{value:.1f} {unit}"

    @staticmethod
    def _format_fraction(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.2f}"

    @staticmethod
    def _format_number(value: float | None, *, precision: int = 1) -> str:
        if value is None:
            return "n/a"
        return f"{value:.{precision}f}"

    @staticmethod
    def _last_action_has_tag(action_history: Sequence[ActionRecord], tag: str) -> bool:
        if not action_history:
            return False
        last_record = action_history[-1]
        return last_record.success and tag in last_record.tags

    @staticmethod
    def _status_for_heart_rate(value: float | None) -> MonitorStatus:
        if value is None:
            return "info"
        if value < 40 or value > 140:
            return "critical"
        if value < 55 or value > 120:
            return "warning"
        return "stable"

    @staticmethod
    def _status_for_map(value: float | None) -> MonitorStatus:
        if value is None:
            return "info"
        if value < 55:
            return "critical"
        if value < 65:
            return "warning"
        return "stable"

    @staticmethod
    def _status_for_spo2(value: float | None) -> MonitorStatus:
        if value is None:
            return "info"
        if value < 0.88:
            return "critical"
        if value < 0.94:
            return "warning"
        return "stable"

    @staticmethod
    def _status_for_respiration(value: float | None) -> MonitorStatus:
        if value is None:
            return "info"
        if value < 8 or value > 35:
            return "critical"
        if value < 10 or value > 28:
            return "warning"
        return "stable"

    @staticmethod
    def _status_for_etco2(value: float | None) -> MonitorStatus:
        if value is None:
            return "info"
        if value < 20 or value > 60:
            return "critical"
        if value < 25 or value > 50:
            return "warning"
        return "stable"

    @staticmethod
    def _status_for_shock_index(value: float | None) -> MonitorStatus:
        if value is None:
            return "info"
        if value > 1.3:
            return "critical"
        if value > 0.9:
            return "warning"
        return "stable"
