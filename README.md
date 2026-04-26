---
title: Pulse-ER Environment
emoji: 🩺
colorFrom: green
colorTo: blue
---

# Pulse-ER: Medical Trauma RL Environment

Pulse-ER is a trauma-management reinforcement learning environment built on top of the Pulse physiology engine. The agent observes real patient state, chooses structured clinical tools, advances simulated time, and is rewarded for stabilizing the patient with the right sequencing of care.

## Problem

Large language models can describe trauma care, but they are weak at acting under competing physiologic priorities. A patient can be hypoxic, hypotensive, and actively bleeding at the same time. Pulse-ER turns that into a measurable environment: the model must decide what to do first, what to delay, and how to react when the patient deteriorates.

## What This Repo Contains

- A real Pulse-backed trauma environment in [server/pulse_physiology_env_environment.py](./server/pulse_physiology_env_environment.py)
- A mock backend for safe GRPO and regression work in [server/adapters.py](./server/adapters.py)
- A frozen tool and observation contract in [tool_catalog.py](./tool_catalog.py) and [models.py](./models.py)
- A reward engine with dense shaping, terminal outcome scoring, and anti-exploitation guards in [server/reward_engine.py](./server/reward_engine.py)
- ATLS judging and monitor payloads for demo use in [server/atls_judge.py](./server/atls_judge.py) and [server/patient_monitor.py](./server/patient_monitor.py)

## Environment

The current consumer-facing contract exposes 17 structured clinical tools, including:

- `get_vitals`
- `advance_time`
- `give_oxygen`
- `give_fluids`
- `control_bleeding`
- `airway_support`
- `needle_decompression`
- `pericardiocentesis`
- `give_pressor`
- `get_blood_gas`
- `get_cbc`
- `get_bmp`

Under the hood, the real runtime also exposes the broader clinical tool catalog used for demo and engine testing.

## Results

The committed figure below is generated from the current mock evaluation harness in `python -m pulse_physiology_env.eval_mock`.

![Reward Curve](./docs/reward_curve.png)

Current policy ranking on the committed mock benchmark:

- `expert`: `4.480`
- `llm_demo`: `3.330`
- `random`: `-3.334`
- `no_action`: `-9.053`

That ordering is enforced by the regression suite.

## Scenarios

The project currently supports:

- Deterministic mock scenarios for training and regression:
  - `baseline_stable`
  - `respiratory_distress`
  - `hemorrhagic_shock`
- Real Pulse-backed trauma scenarios with randomized patient pools:
  - `trauma_easy_soldier`
  - `trauma_medium_carol`
  - `trauma_hard_underweight`
  - `polytrauma_demo`

## Quick Start

Install the Python package from the repo root:

```bash
pip install -e .
```

Run the mock evaluation harness:

```bash
python -m pulse_physiology_env.eval_mock
```

Run the lightweight regression checks:

```bash
python -m pulse_physiology_env.integration_smoke
python -m pulse_physiology_env.smoke_test
```

Run one mock episode:

```bash
python -m pulse_physiology_env.run_mock_episode --backend mock --scenario respiratory_distress --policy expert
```

## Training and Demo Assets

- Demo walkthrough: [DEMO_SCRIPT.md](./DEMO_SCRIPT.md)
- Integration checklist: [INTEGRATION_CHECKPOINT.md](./INTEGRATION_CHECKPOINT.md)
- HF job launcher: [run_hf_job.py](./run_hf_job.py)
- Spec: [SPEC.md](./SPEC.md)

Public Hugging Face Space URL is not committed yet, but the repo already contains the Space-oriented app configuration and Docker path at the repo root.

## Real Runtime Status

The real Pulse runtime has already been swept end-to-end:

- `68` raw real-runtime public tools checked
- `64` executed successfully
- `4` return structured `UNSUPPORTED_BY_ENGINE` in the current local Pulse build:
  - `administer_atropine_bolus`
  - `start_dopamine_infusion`
  - `administer_plasma`
  - `activate_massive_transfusion_protocol`

## Repo Layout

```text
.
├── README.md
├── pyproject.toml
├── models.py
├── patient_state.py
├── tool_catalog.py
├── real_backend.py
├── train_grpo.py
├── run_hf_job.py
├── server/
│   ├── adapters.py
│   ├── app.py
│   ├── pulse_engine_adapter.py
│   ├── pulse_physiology_env_environment.py
│   ├── reward_engine.py
│   └── tools.py
├── engine/
└── engine-build/
```

## Submission Notes

- The mock backend is the safe path for GRPO and notebook demos.
- The real backend is wired to the local Pulse install for physiology-grounded execution.
- The repo is now flattened at the root, with runtime/server code kept under `server/`.
