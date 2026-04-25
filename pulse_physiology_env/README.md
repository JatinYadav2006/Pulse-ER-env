---
title: Pulse-ER Environment Server
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - physiology
  - simulation
---

# Pulse-ER Environment

Pulse-ER is an OpenEnv-compatible physiology environment for training and evaluating clinical reasoning over a simulated patient state.

At the current stage, this folder contains a **hybrid state**:

- a real Pulse-backed runtime path from Person 1
- a mock-side regression and evaluation path from Person 2
- a frozen shared contract that both sides should continue to honor

## Why This Exists

The hackathon product goal is not just to expose raw vitals. It is to create a tool-driven clinical environment where an agent can:

- inspect a patient state
- decide on an intervention
- apply actions over time
- observe deterioration or stabilization
- be scored on the quality of care

This repository is structured so two people can work in parallel:

- **Person 1** owns the real Pulse runtime integration
- **Person 2** owns the consumer-side contract, mock environment, evaluation, and training-facing product layer

## Current Status

Implemented now:

- frozen contract in [SPEC.md](./SPEC.md)
- demo and judging walkthrough in [DEMO_SCRIPT.md](./DEMO_SCRIPT.md)
- real integration checklist in [INTEGRATION_CHECKPOINT.md](./INTEGRATION_CHECKPOINT.md)
- structured models in [models.py](./models.py)
- real Pulse engine adapter in [server/pulse_engine_adapter.py](./server/pulse_engine_adapter.py)
- real tool execution in [server/tools.py](./server/tools.py)
- Pulse-backed OpenEnv environment in [server/pulse_physiology_env_environment.py](./server/pulse_physiology_env_environment.py)
- `MockPulseAdapter` in [server/adapters.py](./server/adapters.py)
- deterministic mock scenarios in [server/mock_scenarios.py](./server/mock_scenarios.py)
- smoke tests in [smoke_test.py](./smoke_test.py)
- reward logic in [rewards.py](./rewards.py)
- policy comparison harness in [eval_mock.py](./eval_mock.py)

Still in progress:

- formal swap from the mock adapter boundary to the real runtime boundary for training-facing loops
- final alignment review between the frozen contract and the richer real runtime state
- integration verification that mock-side consumer code continues to work cleanly against the real path
- submission-facing TRL/Unsloth training artifacts, reward plots, and Hugging Face Space packaging

## Tier Framing

The product is organized around a 3-tier tool story.

### Tier 1: Read and Assess

These are safe information tools used to understand the patient.

- `get_vitals`
- `summarize_state`
- `check_deterioration`

### Tier 2: Intervene

These change patient state and should have visible physiological effects.

- `advance_time`
- `give_oxygen`
- `give_fluids`
- `control_bleeding`
- `position_patient`
- `airway_support`

### Tier 3: Clinical Reasoning Workflows

These sit above raw tools and help the agent act like a clinician.

- `recommend_next_step`
- future compound workflows such as triage summaries, scenario reports, and deterioration response plans

## Frozen Contract

The shared interface between Person 1 and Person 2 lives in [SPEC.md](./SPEC.md).

The key public state fields are:

- `scenario_id`
- `patient_id`
- `sim_time_s`
- `heart_rate_bpm`
- `systolic_bp_mmhg`
- `diastolic_bp_mmhg`
- `spo2`
- `respiration_rate_bpm`
- `blood_volume_ml`
- `mental_status`
- `active_alerts`
- `done`

The key public action shape is:

```json
{
  "tool_name": "give_oxygen",
  "arguments": {
    "flow_lpm": 15
  },
  "reasoning": "Patient is hypoxemic and in respiratory distress."
}
```

## Mock Today, Real Pulse Tomorrow

The current implementation is intentionally split into two backend paths:

- `MockPulseAdapter`: deterministic, testable, used for Person 2 regression and early policy work
- Pulse-backed runtime path: implemented through `PulseEngineAdapter`, `PulseToolExecutor`, and the real OpenEnv environment

The important design rule is:

**Person 2 should be able to move from the mock runtime path to the real Pulse runtime path without rewriting models, parser logic, reward harnesses, or evaluation code.**

That is why the adapter boundary exists in [server/adapters.py](./server/adapters.py).

## Mock Scenarios

The current mock environment includes three deterministic scenarios:

- `baseline_stable`
- `respiratory_distress`
- `hemorrhagic_shock`

These exist to support:

- early policy iteration
- smoke testing
- reward debugging
- demo flow design before the real runtime is ready

The real Pulse-backed runtime scenarios currently live in [server/scenarios.py](./server/scenarios.py).

## Development Workflow

### 1. Run Smoke Tests

This validates that good care beats bad care in the deterministic mock environment.

```bash
python -m pulse_physiology_env.smoke_test
```

### 2. Run Policy Evaluation

This compares:

- expert policy
- llm demo policy
- random policy
- no-action policy

```bash
python -m pulse_physiology_env.eval_mock
```

The expected ranking is:

```text
expert > llm_demo > random > no_action
```

### 3. Run the OpenEnv Server

This requires the OpenEnv dependency stack to be installed.

```bash
uvicorn server.app:app --reload
```

### 4. Submission-Facing GRPO Training

The hackathon judges explicitly want an OpenEnv-native training story built with
TRL or Unsloth, not only a local custom trainer. The submission-facing training
entrypoint is [train_grpo.py](./train_grpo.py), which uses:

- the public [client.py](./client.py) client
- the TRL/OpenEnv-style [trl_env.py](./trl_env.py) environment factory
- GRPO over the running OpenEnv server

Example local launch:

```bash
python -m pulse_physiology_env.train_grpo \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --env-url http://127.0.0.1:8000 \
  --scenario polytrauma_demo
```

The internal [train_online.py](./train_online.py) script is still useful for
fast smoke testing and feature debugging, but it is not the primary
submission-facing training path.

## Backend Integration Contract For Person 1

Person 1 should keep the real Pulse runtime aligned with the existing contract, not invent a new one.

### Required outputs from Person 1

- produce the exact public `PatientState` fields defined in [SPEC.md](./SPEC.md)
- preserve canonical units
- preserve exact tool names
- return the same top-level response envelope
- return structured errors instead of plain failures

### Person 1 should not change without syncing

- field names
- units
- tool names
- enum values for `mental_status`
- top-level response keys

### Minimum handoff expectation

As the real runtime evolves, Person 2 should still be able to keep the following working with minimal code change:

- [models.py](./models.py)
- [client.py](./client.py)
- [smoke_test.py](./smoke_test.py)
- [rewards.py](./rewards.py)
- [eval_mock.py](./eval_mock.py)

## Project Structure

```text
pulse_physiology_env/
|-- README.md
|-- SPEC.md
|-- PERSON1_HANDOFF.md
|-- DEMO_SCRIPT.md
|-- INTEGRATION_CHECKPOINT.md
|-- __init__.py
|-- client.py
|-- models.py
|-- patient_state.py
|-- rewards.py
|-- smoke_test.py
|-- eval_mock.py
|-- generate_seed_trajectories.py
`-- server/
    |-- __init__.py
    |-- adapters.py
    |-- app.py
    |-- mock_scenarios.py
    |-- pulse_engine_adapter.py
    |-- scenarios.py
    |-- tools.py
    |-- pulse_physiology_env_environment.py
    `-- Dockerfile
```

## What This README Is For

This README is meant to help three audiences quickly:

- **teammates**: understand the ownership split and integration contract
- **judges**: understand the product direction and the clinical-tool framing
- **future us**: remember what is mock scaffolding versus what is already real Pulse-backed behavior
