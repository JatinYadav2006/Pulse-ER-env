# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Pulse Physiology Env Environment.

This module creates an HTTP server that exposes the PulsePhysiologyEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

from pydantic import BaseModel, Field
from fastapi import HTTPException

try:
    import openenv.core.env_server.http_server as openenv_http_server
    import openenv.core.env_server.serialization as openenv_serialization
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import PulsePhysiologyObservation, ToolAction
    from .pathology_architect import PathologyArchitect
    from .pulse_physiology_env_environment import PulsePhysiologyEnvironment
except ModuleNotFoundError:
    from models import PulsePhysiologyObservation, ToolAction
    from server.pathology_architect import PathologyArchitect
    from server.pulse_physiology_env_environment import PulsePhysiologyEnvironment


class PathologyGenerationRequest(BaseModel):
    """Request model for generated trauma cases."""

    patient_id: str = Field(..., description="Known baseline patient identifier.")
    injury_type: str = Field(..., description="One of the supported generated injury types.")
    severity: float = Field(..., ge=0.0, le=1.0, description="Severity on a 0-1 scale.")


def _serialize_observation_with_metadata(observation):
    obs_dict = observation.model_dump(exclude={"reward"})
    return {
        "observation": obs_dict,
        "reward": observation.reward,
        "done": observation.done,
    }


openenv_http_server.serialize_observation = _serialize_observation_with_metadata
openenv_serialization.serialize_observation = _serialize_observation_with_metadata
create_app = openenv_http_server.create_app


# Create the app with web interface and README integration
app = create_app(
    PulsePhysiologyEnvironment,
    ToolAction,
    PulsePhysiologyObservation,
    env_name="pulse_physiology_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)
_PATHOLOGY_ARCHITECT = PathologyArchitect()


@app.get("/pathology/library")
def pathology_library() -> dict[str, list[str]]:
    """Expose authoring options for the PathologyArchitect."""

    return {
        "patients": _PATHOLOGY_ARCHITECT.supported_patients(),
        "injury_types": _PATHOLOGY_ARCHITECT.supported_injury_types(),
    }


@app.post("/pathology/generate")
def generate_pathology(request: PathologyGenerationRequest) -> dict[str, object]:
    """Return a generated pathology blueprint that can be sent to reset()."""

    try:
        blueprint = _PATHOLOGY_ARCHITECT.build_blueprint(
            patient_id=request.patient_id,
            injury_type=request.injury_type,
            severity=request.severity,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return blueprint.as_dict()


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m pulse_physiology_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn pulse_physiology_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
