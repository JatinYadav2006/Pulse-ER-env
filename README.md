---
title: Pulse-ER
emoji: "🫀"
colorFrom: red
colorTo: blue
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - physiology
  - trauma-medicine
  - grpo
pinned: true
---

# Pulse-ER — Emergency Response Training Environment

*A physiologically-validated reinforcement learning environment for training agents to manage critical trauma patients during the golden hour of emergency medicine.*

`64 clinical tools` · `20 patient profiles` · `Pulse 4.3.2 validated`

Pulse-ER is a research-grade trauma RL environment built on the **Pulse Physiology Engine 4.3.2**, a validated human physiology simulator used in US military medical training. The environment models emergency decision-making during the golden hour, where survival depends on timing, sequencing, and reassessment rather than one-shot classification. Every intervention is executed against real simulated physiology: hemorrhage changes perfusion, pneumothorax changes oxygenation and mechanics, drugs change hemodynamics, and delay makes recovery harder. The core result is simple: the agent is forced to learn ATLS-style trauma protocol from consequences, not pattern matching.

## Why this environment is hard

This environment is hard because the physiology is real enough to punish shortcuts. Pulse 4.3.2 simulates cardiovascular, respiratory, and blood chemistry dynamics at the organ-system level, so downstream effects emerge from the engine rather than from scripted reward tables.

This environment is also partially observable. Observations can be perturbed by configurable bedside noise, including dropped or noisy SpO2, blood pressure, respiratory rate, and EtCO2. The agent must act under uncertainty instead of reading perfect state.

The final difficulty is clinical sequencing. Several scenarios are built around treatment traps where the obvious-looking action is wrong. In tension pneumothorax, fluids before decompression worsen the patient and are penalized immediately, so the policy must learn protocol order rather than symptom-to-tool mapping.

## Environment design

### Observation space

The environment exposes a stable `PatientState` contract with delayed diagnostics and clinically meaningful derived fields.

| Field group | Key fields |
|---|---|
| Hemodynamics | `heart_rate_bpm`, `systolic_bp_mmhg`, `diastolic_bp_mmhg`, `mean_arterial_pressure_mmhg`, `blood_volume_ml` |
| Respiratory | `spo2`, `respiration_rate_bpm`, `breath_sounds`, `etco2_mmhg` |
| Clinical state | `mental_status`, `shock_index`, `lactate_trend`, `active_alerts`, `scenario_difficulty` |
| Delayed diagnostics | `pending_diagnostics`, `ready_diagnostics`, `abg_result`, `cbc_result`, `bmp_result` |
| Active therapy | `active_infusions`, `active_hemorrhages`, `oxygen_device`, `airway_support` |

Diagnostics are not instant. Labs must be ordered, simulated time must pass, and the completed study must then be retrieved from `ready_diagnostics`.

### Action space

The consumer-facing contract exposes 17 tools across 5 categories:

- Assessment: `get_vitals`, `check_deterioration`, `summarize_state`
- Airway/breathing: `give_oxygen`, `airway_support`, `needle_decompression`
- Circulation: `control_bleeding`, `give_fluids`, `give_pressor`
- Diagnostics: `get_blood_gas`, `get_cbc`, `get_bmp`
- Procedure/time: `perform_pericardiocentesis`, `advance_time`

Internally, the runtime exposes a 64-tool engine-backed clinical surface. Four tools are explicitly unavailable in the local Pulse build because the required substance files are missing: atropine, dopamine, plasma, and MTP. Those actions return structured `UNSUPPORTED_BY_ENGINE` instead of crashing.

## Reward engine formula

```text
R_t = 0.35 × MAP_stability
    + 0.25 × SpO2_efficiency  
    + 0.20 × lactate_trend
    + 0.10 × intervention_safety
    + 0.10 × diagnostic_timeliness
    + R_terminal (on episode end)
```

`MAP_stability` rewards restoration of perfusion. `SpO2_efficiency` rewards meaningful oxygenation improvement, not just action spam. `lactate_trend` tracks whether shock is actually reversing.

`intervention_safety` applies hard order-sensitive penalties, including fluids before decompression (`-0.8`), pressors before volume (`-0.5`), and succinylcholine without a secured airway path (`-1.0`). `diagnostic_timeliness` rewards early studies and correct retrieval of delayed results.

The terminal term includes survival bonus, time efficiency, sequence quality, and difficulty scaling. Anti-exploitation guards penalize repeated tool spam and neglected ready diagnostics.

## Environment design

### Observation space

### Time pressure mechanic

After three minutes of simulated time without stabilization, a deterioration multiplier activates and increases at `0.15` per minute per severity unit. At the same time, intervention effectiveness decays. The environment therefore teaches that hesitation is not neutral.

## Patient profiles

The patient corpus is a measured result, not a cosmetic feature. Twenty baseline Pulse profiles were run through a standardized trauma challenge and ranked by observed resilience using post-insult MAP, SpO2, shock index, mental status, and short no-intervention survival.

| Tier | Patients | Characteristics |
|---|---|---|
| Easy (7) | Bradycardic, Nathan, StandardMale, DefaultMale, Overweight, Carol, Jeff | Higher baseline cardiovascular reserve, tolerated standardized trauma challenge |
| Medium (7) | Jane, Cynthia, Underweight, DefaultFemale, Rick, Soldier, ExtremeMale | Moderate resilience, meaningful intervention required |
| Hard (6) | StandardFemale, Joel, Tachycardic, ExtremeFemale, Gus, Hassan | Most fragile under trauma insult, smallest intervention window |

Several assignments are intentionally counterintuitive. Bradycardic appears in easy and StandardFemale appears in hard because the classification is data-driven from measured physiology rather than patient naming.

## The three golden scenarios

### Scenario 1: Class III hemorrhagic shock

| Item | Value |
|---|---|
| Injuries | Single compartment hemorrhage, `150 mL/min` |
| Correct path | tourniquet → crystalloid → norepinephrine |
| Teaching point | volume before pressors |
| Survival window | `8` simulated minutes |

### Scenario 2: Tension pneumothorax masquerading as shock (DEMO SCENARIO)

| Item | Value |
|---|---|
| Injuries | Abdominal hemorrhage (`80 mL/min`) + left tension pneumothorax |
| Trap | fluids worsen patient — must decompress first |
| Correct path | auscultate → POCUS → needle decompression → crystalloid → norepinephrine |
| Teaching point | diagnose before treating |
| Survival window | `6` simulated minutes |
| Demo moment | naive agent dies, trained agent survives |

This is the demo case because the physiology is visible and non-scripted. A naive sequence gives fluids into unresolved obstructive physiology and the patient dies. A decompression-first sequence produces the characteristic Pulse response, with SpO2 rising from `0.84` to `0.99`.

### Scenario 3: Cardiac tamponade after penetrating chest trauma

| Item | Value |
|---|---|
| Injuries | Pericardial effusion (`severity 0.7`) + thoracic hemorrhage |
| Trap | Beck's triad — fluid resuscitation minimally effective |
| Correct path | POCUS cardiac → pericardiocentesis → crystalloid |
| Teaching point | obstructive shock requires mechanical relief |
| Survival window | `5` simulated minutes |

## Adversarial evaluation system

The adversarial system measures robustness rather than just average reward. For each of the 20 patients, the injury-stacking adversary runs a fixed combo ladder and records the first combination the agent cannot survive.

1. `tension_pneumothorax`
2. `hemorrhagic_shock`
3. `cardiac_tamponade`
4. `tension_pneumothorax + hemorrhagic_shock`
5. `hemorrhagic_shock + cardiac_tamponade`
6. `tension_pneumothorax + hemorrhagic_shock + cardiac_tamponade`

### Key findings

| Result | Value |
|---|---|
| Generated resets | `120/120` succeeded across all 20 patients and all 6 combos |
| Expert survival on hemorrhage + tamponade | `7/20` at severity `0.7` |
| Expert survival on triple threat | `0/20` at severity `0.7` |
| Threshold representation | `breaking_combo and breaking_severity` |
| Reset handling | automatic severity backoff in `0.1` steps if a combo is terminal at reset |

Hassan is a representative case. That patient survived all three single-injury scenarios and the pneumo-plus-hemorrhage double, but failed on hemorrhage plus tamponade. Clinically, that failure is meaningful because simultaneous active bleeding and obstructive shock create a treatment conflict with no clean sequential ATLS pathway.

## ATLS judge

Every observation includes a human-readable ATLS score from `atls_judge.py`. The judge uses action history plus patient state progression to produce a `0–100` score with readable pass/fail checks.

```text
ATLS Score: 96/100 — Textbook ATLS protocol
✓ PASS  Assessed before treating
✓ PASS  Decompressed before fluids  
✓ PASS  Hemorrhage controlled early
✓ PASS  Labs ordered timely
```

```text
ATLS Score: 14/100 — Critical protocol failure
✗ FAIL  Assessed before treating
✗ FAIL  Decompressed before fluids
✗ FAIL  Hemorrhage controlled early
✓ PASS  No dangerous drug interactions
```

CPR is judged as valid when arrest is present in the patient state history, not only when arrest was manually induced. That covers physiological arrest from deterioration as well as scripted authoring events.

## PathologyArchitect

New cases can be generated on the fly through the PathologyArchitect. It takes `(patient_id, injury_type, severity)` and returns a valid scenario blueprint consumable by the environment.

| Endpoint | Purpose |
|---|---|
| `GET /pathology/library` | list supported patients and injury families |
| `POST /pathology/generate` | generate a scenario blueprint |

Supported injury types:

- `tension_pneumothorax`
- `hemorrhagic_shock`
- `cardiac_tamponade`
- `polytrauma`

## Training

```bash
hf jobs run \
  --with trl \
  --flavor t4-small \
  --env PULSE_ENV_URL=https://your-space.hf.space \
  -- python train_grpo.py
```

The training stack uses GRPO through TRL. Submission-facing runs use Qwen2.5-3B-Instruct with LoRA rank `16`, while mock runs remain the fast iteration path and the real Pulse backend remains the validated evaluation path. The same reward formula above is used during training, so clinical sequencing is part of optimization rather than a post-hoc judge overlay.

### Verified policy ranking

| Policy | Outcome |
|---|---|
| `expert` | positive reward on all scenarios |
| `llm_demo` | positive on easy, negative on hard |
| `random` | `patient_death` on `3/4` real scenarios |
| `no_action` | `patient_death` on `3/4` real scenarios |

## Quick start

```bash
git clone https://github.com/KumarChad/pulse-phisiology-env
cd pulse-phisiology-env

# Install dependencies
pip install -e .

# Run smoke test (mock backend, no Pulse required)
python -m pulse_physiology_env.eval_mock

# Run with real Pulse engine (requires local build)
export PULSE_INSTALL_DIR=/path/to/engine-build/install
python -m pulse_physiology_env.smoke_test

# Run a demo episode
python -m pulse_physiology_env.run_mock_episode \
  --scenario respiratory_distress \
  --policy expert \
  --observation-noise-level 0.3 \
  --time-pressure
```

## Architecture

The codebase is split so training, simulation, and evaluation can evolve without contract drift.

| File | Responsibility |
|---|---|
| `pulse_engine_adapter.py` | Pulse engine interaction, state synthesis, semantic operations |
| `tools.py` | tool registry and clinical tool handlers |
| `reward_engine.py` | dense rewards, terminal rewards, sequence scoring, safety penalties |
| `atls_judge.py` | human-readable protocol scoring |
| `patient_monitor.py` | structured monitor payload for visualization |
| `pathology_architect.py` | generated scenario authoring |
| `scenarios.py` | data-driven patient pools and scenario registry |
| `injury_stack_adversary.py` | adversarial evaluation system |
| `adapters.py` | mock backend with full 17-tool contract |
| `app.py` | FastAPI server with reset/step/health/pathology endpoints |
| `train_grpo.py` | GRPO training entrypoint |

## Research findings

The following results were produced by running the environment against the 20-patient corpus with the standardized trauma protocol.

| Finding | Result |
|---|---|
| Policy separation | expert reward `8.33` on `hemorrhagic_shock` vs random `-17.15` and no_action `-17.10` |
| Adversarial breaking points | `7/20` patients survived double-threat, `0/20` survived triple-threat |
| Difficulty validation | hard patients: MAP `41–59`, SpO2 `0.62–0.83`; easy patients: MAP `~90s`, SpO2 `~0.95–0.96` |
| Reward signal quality | naive pneumo `-0.838`, decompression-first `-0.068` on same patient and seed |

## Limitations and future work

Current limitations:

- 4 tools are unsupported due to missing substance files in the local Pulse 4.3.2 build: atropine, dopamine, plasma, and MTP. These return structured `UNSUPPORTED_BY_ENGINE`.
- `position_patient` is context-only because this build does not expose a native Pulse position action.
- The triple-threat combo is universally lethal at severity `0.7` for the current trained agent and therefore remains an unsolved benchmark level.

Future work:

- severity-escalation adversary layered on top of injury stacking to recover per-patient breaking severity by binary search
- ventilator weaning and prolonged-care scenarios beyond the golden hour
- multi-injury complication events grounded in validated physiology, including rebound pneumothorax and transfusion reactions
- larger-model training runs with the full 64-tool catalog exposed
