# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pulse Physiology Env package exports."""

from .models import (
    EnvironmentResponse,
    ObservationMetadata,
    PulsePhysiologyAction,
    PulsePhysiologyObservation,
    ToolAction,
    ToolError,
    ToolResult,
)
from .patient_state import (
    ArterialBloodGasResult,
    BasicMetabolicPanelResult,
    CompleteBloodCountResult,
    LactateTrend,
    MentalStatus,
    PatientState,
    ScenarioDifficulty,
)
from .tool_catalog import EXTENDED_TOOL_NAMES, INITIAL_TOOL_NAMES, KNOWN_TOOL_NAMES

__all__ = [
    "ArterialBloodGasResult",
    "BasicMetabolicPanelResult",
    "CompleteBloodCountResult",
    "EXTENDED_TOOL_NAMES",
    "EnvironmentResponse",
    "INITIAL_TOOL_NAMES",
    "KNOWN_TOOL_NAMES",
    "LactateTrend",
    "MentalStatus",
    "ObservationMetadata",
    "PatientState",
    "ScenarioDifficulty",
    "PulsePhysiologyAction",
    "PulsePhysiologyObservation",
    "ToolAction",
    "ToolError",
    "ToolResult",
]


def _optional_export(name: str, import_fn) -> None:
    """Import an optional package-level symbol without breaking lightweight consumers."""

    try:
        globals()[name] = import_fn()
    except Exception:  # pragma: no cover - optional runtime/training helpers may not import everywhere
        globals()[name] = None
    else:
        __all__.append(name)


_optional_export("PulsePhysiologyEnv", lambda: __import__(__name__ + ".client", fromlist=["PulsePhysiologyEnv"]).PulsePhysiologyEnv)
_optional_export("RealPulseBackend", lambda: __import__(__name__ + ".real_backend", fromlist=["RealPulseBackend"]).RealPulseBackend)
_optional_export(
    "InjuryStackAdversary",
    lambda: __import__(__name__ + ".injury_stack_adversary", fromlist=["InjuryStackAdversary"]).InjuryStackAdversary,
)
_optional_export("PulseGymEnv", lambda: __import__(__name__ + ".gym_env", fromlist=["PulseGymEnv"]).PulseGymEnv)
