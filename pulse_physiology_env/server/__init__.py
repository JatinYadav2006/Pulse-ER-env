# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pulse Physiology Env environment server components."""

__all__: list[str] = []


def _optional_export(name: str, import_fn) -> None:
    """Import an optional server-side symbol without breaking lightweight consumers."""

    try:
        globals()[name] = import_fn()
    except Exception:  # pragma: no cover - optional runtime helpers may not import everywhere
        globals()[name] = None
    else:
        __all__.append(name)


_optional_export("ATLSJudge", lambda: __import__(__name__ + ".atls_judge", fromlist=["ATLSJudge"]).ATLSJudge)
_optional_export(
    "PathologyArchitect",
    lambda: __import__(__name__ + ".pathology_architect", fromlist=["PathologyArchitect"]).PathologyArchitect,
)
_optional_export(
    "PatientMonitorVisualization",
    lambda: __import__(__name__ + ".patient_monitor", fromlist=["PatientMonitorVisualization"]).PatientMonitorVisualization,
)
_optional_export(
    "PulseEngineAdapter",
    lambda: __import__(__name__ + ".pulse_engine_adapter", fromlist=["PulseEngineAdapter"]).PulseEngineAdapter,
)
_optional_export(
    "PulsePhysiologyEnvironment",
    lambda: __import__(
        __name__ + ".pulse_physiology_env_environment",
        fromlist=["PulsePhysiologyEnvironment"],
    ).PulsePhysiologyEnvironment,
)
_optional_export("RewardEngine", lambda: __import__(__name__ + ".reward_engine", fromlist=["RewardEngine"]).RewardEngine)
_optional_export(
    "PulseToolExecutor",
    lambda: __import__(__name__ + ".tools", fromlist=["PulseToolExecutor"]).PulseToolExecutor,
)
