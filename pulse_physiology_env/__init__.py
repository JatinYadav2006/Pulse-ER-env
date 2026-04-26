# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pulse Physiology Env Environment."""

try:
    from .client import PulsePhysiologyEnv
except ImportError:  # pragma: no cover - enables mock-side work without openenv installed
    PulsePhysiologyEnv = None

try:
    from .real_backend import RealPulseBackend
except ImportError:  # pragma: no cover - allows consumer-side imports without runtime deps
    RealPulseBackend = None

try:
    from .injury_stack_adversary import InjuryStackAdversary
except ImportError:  # pragma: no cover - allows imports when real runtime deps are unavailable
    InjuryStackAdversary = None

from .models import (
    EnvironmentResponse,
    ObservationMetadata,
    PulsePhysiologyAction,
    PulsePhysiologyObservation,
    ToolAction,
    ToolError,
    ToolResult,
)
from .tool_catalog import EXTENDED_TOOL_NAMES, INITIAL_TOOL_NAMES, KNOWN_TOOL_NAMES
from .patient_state import (
    ArterialBloodGasResult,
    BasicMetabolicPanelResult,
    CompleteBloodCountResult,
    LactateTrend,
    MentalStatus,
    PatientState,
    ScenarioDifficulty,
)

__all__ = [
    "ArterialBloodGasResult",
    "BasicMetabolicPanelResult",
    "CompleteBloodCountResult",
    "EXTENDED_TOOL_NAMES",
    "EnvironmentResponse",
    "InjuryStackAdversary",
    "INITIAL_TOOL_NAMES",
    "KNOWN_TOOL_NAMES",
    "LactateTrend",
    "MentalStatus",
    "ObservationMetadata",
    "PatientState",
    "ScenarioDifficulty",
    "PulsePhysiologyAction",
    "PulsePhysiologyObservation",
    "RealPulseBackend",
    "ToolAction",
    "ToolError",
    "ToolResult",
]

if PulsePhysiologyEnv is not None:
    __all__.append("PulsePhysiologyEnv")
