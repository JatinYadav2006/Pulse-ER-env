# Pulse-ER: Medical Trauma RL Environment

Pulse-ER is a trauma-management reinforcement learning environment built on top of the Pulse physiology engine. The agent observes real patient state, chooses structured clinical tools, advances simulated time, and is rewarded for stabilizing the patient with the right sequencing of care.

## Problem

Large language models can describe trauma care, but they are weak at acting under competing physiologic priorities. A patient can be hypoxic, hypotensive, and actively bleeding at the same time. Pulse-ER turns that into a measurable environment: the model must decide what to do first, what to delay, and how to react when the patient deteriorates.

## What This Repo Contains

- A real Pulse-backed trauma environment in [pulse_physiology_env/server/pulse_physiology_env_environment.py](./pulse_physiology_env/server/pulse_physiology_env_environment.py)
- A mock backend for safe GRPO and regression work in [pulse_physiology_env/server/adapters.py](./pulse_physiology_env/server/adapters.py)
- A frozen tool and observation contract in [pulse_physiology_env/tool_catalog.py](./pulse_physiology_env/tool_catalog.py) and [pulse_physiology_env/models.py](./pulse_physiology_env/models.py)
- A reward engine with dense shaping, terminal outcome scoring, and anti-exploitation guards in [pulse_physiology_env/server/reward_engine.py](./pulse_physiology_env/server/reward_engine.py)
- ATLS judging and monitor payloads for demo use in [pulse_physiology_env/server/atls_judge.py](./pulse_physiology_env/server/atls_judge.py) and [pulse_physiology_env/server/patient_monitor.py](./pulse_physiology_env/server/patient_monitor.py)

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

Install the Python package from the environment folder:

```bash
cd pulse_physiology_env
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

- Notebook: [notebooks/Pulse_ER_Quickstart.ipynb](./notebooks/Pulse_ER_Quickstart.ipynb)
- Environment package README / Space-ready app config: [pulse_physiology_env/README.md](./pulse_physiology_env/README.md)
- Demo walkthrough: [pulse_physiology_env/DEMO_SCRIPT.md](./pulse_physiology_env/DEMO_SCRIPT.md)
- Integration checklist: [pulse_physiology_env/INTEGRATION_CHECKPOINT.md](./pulse_physiology_env/INTEGRATION_CHECKPOINT.md)

Public Hugging Face Space URL is not committed yet, but the repo already contains the Space-oriented app configuration and Docker path in the package folder.

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
├── docs/
├── notebooks/
├── pulse_physiology_env/
│   ├── README.md
│   ├── SPEC.md
│   ├── models.py
│   ├── patient_state.py
│   ├── tool_catalog.py
│   ├── real_backend.py
│   └── server/
│       ├── adapters.py
│       ├── app.py
│       ├── pulse_engine_adapter.py
│       ├── pulse_physiology_env_environment.py
│       ├── reward_engine.py
│       └── tools.py
├── engine/
└── engine-build/
```

## Submission Notes

- The mock backend is the safe path for GRPO and notebook demos.
- The real backend is wired to the local Pulse install for physiology-grounded execution.
- The root README is submission-focused; the package README is implementation-focused.
