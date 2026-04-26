"""TRL/OpenEnv environment-factory wrapper for Pulse-ER training.

This module supports two training modes:

- ``real``: uses the public Pulse-ER client and respects client/server
  separation. This is the submission-facing path.
- ``mock``: uses the deterministic mock adapter for tiny local GRPO smoke runs.
  This is an internal safety rail so we can validate training loops quickly
  before burning time on the real Pulse runtime.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from .client import PulsePhysiologyEnv
from .models import PulsePhysiologyAction, PulsePhysiologyObservation


DEFAULT_ENV_URL = "http://127.0.0.1:8000"
DEFAULT_SCENARIO_ID = "polytrauma_demo"
ENV_URL = DEFAULT_ENV_URL
SCENARIO_ID = DEFAULT_SCENARIO_ID
BACKEND_KIND = "real"


def configure_trl_env(
    *,
    env_url: str | None = None,
    scenario_id: str | None = None,
    backend_kind: str | None = None,
) -> None:
    """Update the module-level configuration used by ``PulseToolEnv``.

    TRL's ``environment_factory`` expects a zero-argument class constructor, so
    submission-facing configuration is captured at module level before the
    trainer instantiates environments.
    """

    global BACKEND_KIND, ENV_URL, SCENARIO_ID
    if env_url:
        ENV_URL = env_url
    if scenario_id:
        SCENARIO_ID = scenario_id
    if backend_kind:
        BACKEND_KIND = backend_kind


class PulseToolEnv:
    """Client-backed OpenEnv wrapper exposing Pulse-ER tools as public methods."""

    def __init__(self) -> None:
        self.client = PulsePhysiologyEnv(base_url=ENV_URL)
        self.reward = 0.0
        self.done = False
        self.last_observation: PulsePhysiologyObservation | None = None
        self.last_tool_result: str | None = None
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

    def _run_loop(self) -> None:
        """Own a dedicated event loop for the lifetime of this environment."""

        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _call_client_async(self, method_name: str, *args, **kwargs):
        """Execute one async client call on the dedicated event loop."""

        method = getattr(self.client, method_name)
        return await method(*args, **kwargs)

    def _run_client_call(self, method_name: str, *args, **kwargs):
        """Bridge the async OpenEnv client into TRL's sync environment API."""

        future = asyncio.run_coroutine_threadsafe(
            self._call_client_async(method_name, *args, **kwargs),
            self._loop,
        )
        return future.result()

    def __del__(self) -> None:
        """Best-effort cleanup for the background event loop and websocket client."""

        loop = getattr(self, "_loop", None)
        if loop is None or loop.is_closed():
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self.client.close(), loop)
            future.result(timeout=5)
        except Exception:
            pass
        finally:
            loop.call_soon_threadsafe(loop.stop)

    def reset(self, **kwargs) -> str:
        """Reset the remote environment and return the initial clinical summary."""

        reset_kwargs = dict(kwargs)
        scenario_id = str(reset_kwargs.pop("scenario_id", None) or SCENARIO_ID)
        result = self._run_client_call("reset", scenario_id=scenario_id, **reset_kwargs)
        self.reward = float(result.reward or 0.0)
        self.done = bool(result.done)
        self.last_observation = result.observation
        self.last_tool_result = None
        return self._format_observation(result.observation)

    def get_vitals(self) -> str:
        """Read the current bedside vital signs.

        Returns:
            A concise vital-sign summary from the environment.
        """

        return self._invoke("get_vitals")

    def advance_time(self, seconds: float = 30.0) -> str:
        """Advance the simulation clock.

        Args:
            seconds: Number of simulated seconds to advance.

        Returns:
            The environment's update after time advances.
        """

        return self._invoke("advance_time", seconds=seconds)

    def give_oxygen(
        self,
        flow_lpm: float = 15.0,
        device: str = "non_rebreather_mask",
        monitor_seconds: float = 60.0,
    ) -> str:
        """Provide supplemental oxygen.

        Args:
            flow_lpm: Oxygen flow rate in liters per minute.
            device: Delivery device such as ``nasal_cannula`` or ``non_rebreather_mask``.
            monitor_seconds: Optional monitoring window after the intervention.

        Returns:
            The environment's post-intervention tool response.
        """

        return self._invoke(
            "give_oxygen",
            flow_lpm=flow_lpm,
            device=device,
            monitor_seconds=monitor_seconds,
        )

    def give_fluids(
        self,
        volume_ml: float = 500.0,
        fluid_type: str = "blood",
        rate_ml_per_min: float = 150.0,
        monitor_seconds: float = 60.0,
    ) -> str:
        """Administer IV fluids or blood products.

        Args:
            volume_ml: Infused volume in milliliters.
            fluid_type: Fluid or blood product type.
            rate_ml_per_min: Infusion rate in milliliters per minute.
            monitor_seconds: Optional monitoring window after the intervention.

        Returns:
            The environment's post-intervention tool response.
        """

        return self._invoke(
            "give_fluids",
            volume_ml=volume_ml,
            fluid_type=fluid_type,
            rate_ml_per_min=rate_ml_per_min,
            monitor_seconds=monitor_seconds,
        )

    def control_bleeding(
        self,
        site: str = "right_leg",
        method: str = "tourniquet",
        monitor_seconds: float = 60.0,
    ) -> str:
        """Apply hemorrhage control.

        Args:
            site: Active hemorrhage site to target.
            method: Bleeding-control method such as ``tourniquet`` or ``pressure``.
            monitor_seconds: Optional monitoring window after the intervention.

        Returns:
            The environment's post-intervention tool response.
        """

        return self._invoke(
            "control_bleeding",
            site=site,
            method=method,
            monitor_seconds=monitor_seconds,
        )

    def position_patient(self, position: str = "supine") -> str:
        """Reposition the patient.

        Args:
            position: Target position such as ``supine`` or ``upright``.

        Returns:
            The environment's post-positioning response.
        """

        return self._invoke("position_patient", position=position)

    def airway_support(
        self,
        support_type: str = "cpap",
        monitor_seconds: float = 60.0,
    ) -> str:
        """Provide airway support.

        Args:
            support_type: Airway support mode such as ``cpap`` or ``bag_valve_mask``.
            monitor_seconds: Optional monitoring window after the intervention.

        Returns:
            The environment's post-intervention tool response.
        """

        return self._invoke(
            "airway_support",
            support_type=support_type,
            monitor_seconds=monitor_seconds,
        )

    def summarize_state(self) -> str:
        """Summarize the current clinical state.

        Returns:
            A concise state summary from the environment.
        """

        return self._invoke("summarize_state")

    def check_deterioration(self) -> str:
        """Check whether the patient is worsening.

        Returns:
            The environment's deterioration assessment.
        """

        return self._invoke("check_deterioration")

    def recommend_next_step(self) -> str:
        """Ask the environment for the most appropriate next action.

        Returns:
            The environment's recommended next step.
        """

        return self._invoke("recommend_next_step")

    def give_pressor(
        self,
        pressor: str = "norepinephrine",
        rate_ml_per_min: float = 7.5,
        monitor_seconds: float = 60.0,
    ) -> str:
        """Start or titrate a vasopressor infusion.

        Args:
            pressor: Pressor agent name.
            rate_ml_per_min: Infusion rate in milliliters per minute.
            monitor_seconds: Optional monitoring window after the change.

        Returns:
            The environment's post-intervention tool response.
        """

        return self._invoke(
            "give_pressor",
            pressor=pressor,
            rate_ml_per_min=rate_ml_per_min,
            monitor_seconds=monitor_seconds,
        )

    def needle_decompression(
        self,
        side: str = "left",
        monitor_seconds: float = 90.0,
    ) -> str:
        """Perform needle decompression.

        Args:
            side: Chest side to decompress.
            monitor_seconds: Optional monitoring window after decompression.

        Returns:
            The environment's post-intervention tool response.
        """

        return self._invoke(
            "needle_decompression",
            side=side,
            monitor_seconds=monitor_seconds,
        )

    def pericardiocentesis(
        self,
        rate_ml_per_min: float = 10.0,
        monitor_seconds: float = 90.0,
    ) -> str:
        """Perform pericardiocentesis when tamponade is suspected.

        Args:
            rate_ml_per_min: Drainage rate in milliliters per minute.
            monitor_seconds: Optional monitoring window after the intervention.

        Returns:
            The environment's post-intervention tool response.
        """

        return self._invoke(
            "pericardiocentesis",
            rate_ml_per_min=rate_ml_per_min,
            monitor_seconds=monitor_seconds,
        )

    def get_respiratory_status(self) -> str:
        """Read a respiratory-focused bedside summary.

        Returns:
            A concise respiratory status summary from the environment.
        """

        return self._invoke("get_respiratory_status")

    def get_blood_gas(self) -> str:
        """Order or review arterial blood gas results.

        Returns:
            The environment's blood gas report.
        """

        return self._invoke("get_blood_gas")

    def get_cbc(self) -> str:
        """Order or review complete blood count results.

        Returns:
            The environment's CBC report.
        """

        return self._invoke("get_cbc")

    def get_bmp(self) -> str:
        """Order or review basic metabolic panel results.

        Returns:
            The environment's BMP report.
        """

        return self._invoke("get_bmp")

    def _invoke(self, tool_name: str, **arguments: Any) -> str:
        """Execute one tool call and convert the environment result into text."""

        if self.done:
            raise ValueError("Game over.")

        action = PulsePhysiologyAction(tool_name=tool_name, arguments=arguments)
        result = self._run_client_call("step", action)
        self.reward = float(result.reward or 0.0)
        self.done = bool(result.done)
        self.last_observation = result.observation

        tool_result = getattr(result.observation, "tool_result", None)
        error = getattr(result.observation, "error", None)
        if error is not None:
            message = f"{error.code}: {error.message}"
            self.last_tool_result = message
            if self.done:
                raise ValueError(message)
            return message

        if tool_result is not None and tool_result.message:
            self.last_tool_result = tool_result.message
            return tool_result.message

        fallback = self._format_observation(result.observation)
        self.last_tool_result = fallback
        return fallback

    @staticmethod
    def _format_observation(observation: PulsePhysiologyObservation) -> str:
        """Render a concise text observation for the model."""

        spo2 = "unknown" if observation.spo2 is None else f"{observation.spo2 * 100:.1f}%"
        systolic = "?" if observation.systolic_bp_mmhg is None else f"{observation.systolic_bp_mmhg:.1f}"
        diastolic = "?" if observation.diastolic_bp_mmhg is None else f"{observation.diastolic_bp_mmhg:.1f}"
        heart_rate = "?" if observation.heart_rate_bpm is None else f"{observation.heart_rate_bpm:.1f}"
        resp_rate = "?" if observation.respiration_rate_bpm is None else f"{observation.respiration_rate_bpm:.1f}"
        alerts = ", ".join(observation.active_alerts) if observation.active_alerts else "none"
        mental_status = getattr(observation.mental_status, "value", observation.mental_status)
        return (
            f"Scenario={observation.scenario_id}; "
            f"HR={heart_rate} bpm; "
            f"BP={systolic}/{diastolic} mmHg; "
            f"SpO2={spo2}; "
            f"RR={resp_rate}; "
            f"MentalStatus={mental_status}; "
            f"Alerts={alerts}."
        )


class MockPulseToolEnv:
    """Deterministic mock training wrapper for tiny internal GRPO smoke runs."""

    def __init__(self) -> None:
        from .server.adapters import MockPulseAdapter

        self.backend = MockPulseAdapter(default_scenario_id=SCENARIO_ID)
        self.reward = 0.0
        self.done = False
        self.last_observation: PulsePhysiologyObservation | None = None
        self.last_tool_result: str | None = None

    def reset(self, **kwargs) -> str:
        """Reset the mock environment and return the initial clinical summary."""

        reset_kwargs = dict(kwargs)
        scenario_id = str(reset_kwargs.pop("scenario_id", None) or SCENARIO_ID)
        response = self.backend.reset(scenario_id, **reset_kwargs)
        self.reward = float(response.reward or 0.0)
        self.done = bool(response.done)
        self.last_observation = response.observation
        self.last_tool_result = None
        return PulseToolEnv._format_observation(response.observation)

    def get_vitals(self) -> str:
        """Read the current bedside vital signs.

        Returns:
            A concise vital-sign summary from the environment.
        """

        return self._invoke("get_vitals")

    def advance_time(self, seconds: float = 30.0) -> str:
        """Advance the simulation clock.

        Args:
            seconds: Number of simulated seconds to advance.

        Returns:
            The environment's update after time advances.
        """

        return self._invoke("advance_time", seconds=seconds)

    def give_oxygen(self, flow_lpm: float = 15.0) -> str:
        """Provide supplemental oxygen.

        Args:
            flow_lpm: Oxygen flow rate in liters per minute.

        Returns:
            The environment's post-intervention tool response.
        """

        return self._invoke("give_oxygen", flow_lpm=flow_lpm)

    def give_fluids(self, volume_ml: float = 500.0) -> str:
        """Administer IV fluids.

        Args:
            volume_ml: Infused volume in milliliters.

        Returns:
            The environment's post-intervention tool response.
        """

        return self._invoke("give_fluids", volume_ml=volume_ml)

    def control_bleeding(self) -> str:
        """Apply bleeding control measures.

        Returns:
            The environment's post-intervention tool response.
        """

        return self._invoke("control_bleeding")

    def position_patient(self, position: str = "supine") -> str:
        """Reposition the patient.

        Args:
            position: Target position such as ``supine`` or ``upright``.

        Returns:
            The environment's post-positioning response.
        """

        return self._invoke("position_patient", position=position)

    def airway_support(self, mode: str = "basic") -> str:
        """Provide airway support.

        Args:
            mode: Airway support mode.

        Returns:
            The environment's post-intervention tool response.
        """

        return self._invoke("airway_support", mode=mode)

    def summarize_state(self) -> str:
        """Summarize the current clinical state.

        Returns:
            A concise state summary from the environment.
        """

        return self._invoke("summarize_state")

    def check_deterioration(self) -> str:
        """Check whether the patient is worsening.

        Returns:
            The environment's deterioration assessment.
        """

        return self._invoke("check_deterioration")

    def recommend_next_step(self) -> str:
        """Ask the environment for the most appropriate next action.

        Returns:
            The environment's recommended next step.
        """

        return self._invoke("recommend_next_step")

    def _invoke(self, tool_name: str, **arguments: Any) -> str:
        """Execute one mock tool call and return human-readable feedback."""

        if self.done:
            raise ValueError("Game over.")

        response = self.backend.step(PulsePhysiologyAction(tool_name=tool_name, arguments=arguments))
        self.reward = float(response.reward or 0.0)
        self.done = bool(response.done)
        self.last_observation = response.observation

        if response.error is not None:
            message = f"{response.error.code}: {response.error.message}"
            self.last_tool_result = message
            if self.done:
                raise ValueError(message)
            return message

        if response.tool_result is not None and response.tool_result.message:
            self.last_tool_result = response.tool_result.message
            return response.tool_result.message

        fallback = PulseToolEnv._format_observation(response.observation)
        self.last_tool_result = fallback
        return fallback


def get_environment_factory():
    """Return the configured TRL environment factory class."""

    return MockPulseToolEnv if BACKEND_KIND == "mock" else PulseToolEnv
