---
title: Pulse-ER
emoji: "🫀"
colorFrom: red
colorTo: blue
---

# Pulse-ER — Emergency Response Training Environment

*A physiologically-validated reinforcement learning environment for training agents to manage critical trauma patients during the golden hour of emergency medicine.*

`64 clinical tools` · `20 patient profiles` · `Pulse 4.3.2 validated`

Pulse-ER is a reinforcement learning environment built on the **Pulse Physiology Engine 4.3.2**, a validated human physiology simulator used in US military medical training. It places an agent inside the first hour of trauma care, where treatment quality depends on sequencing, uncertainty, and speed rather than single-step classification. Unlike toy medical environments that reward pattern matching against fixed templates, Pulse-ER grounds every intervention in a real organ-system simulation of hemorrhage, oxygenation, ventilation, perfusion, and shock. The result is an environment where clinical actions have downstream physiological consequences, delayed treatment becomes less effective, and the central question is whether an agent can learn trauma protocol under pressure rather than merely imitate symptoms.

## Why this environment is hard

This environment matters because the agent is solving a genuinely coupled physiology problem rather than a symbolic game. Pulse-ER is backed by Pulse 4.3.2, a validated C++ physiology engine that simulates drug pharmacokinetics, hemorrhage dynamics, respiratory mechanics, and cardiac function at the organ-system level. The agent cannot hallucinate physiology or invent shortcuts: fluids change preload, decompression changes ventilation-perfusion dynamics, and untreated bleeding continues to erode perfusion in simulated time.

This environment is also partially observable in the way real emergency care is partially observable. Observations can be perturbed by configurable measurement noise, including reading fluctuations and dropped measurements from SpO2, blood pressure, respiratory rate, and EtCO2. The agent therefore acts on uncertain bedside data rather than on clean ground-truth state, which makes reassessment and diagnostic timing part of the task rather than mere interface decoration.

The hardest challenge is sequential decision-making under clinical traps. Several cases are constructed so that the intuitively obvious action is wrong: in tension pneumothorax masquerading as shock, giving fluids before decompression worsens the patient and is penalized immediately. The reward shaping is intentionally sequence-aware, so the environment teaches ATLS ordering and not just symptom-to-drug lookup.

## Environment design

The environment is designed around a stable `PatientState` contract so the same clinical episode can be consumed by a training loop, a tool-using policy, or a judge-facing interface without rewriting the schema. Key fields include `heart_rate_bpm`, `systolic_bp_mmhg`, `diastolic_bp_mmhg`, `mean_arterial_pressure_mmhg`, `spo2`, `respiration_rate_bpm`, `blood_volume_ml`, `mental_status`, `shock_index`, `lactate_trend`, `breath_sounds`, `active_alerts`, `pending_diagnostics`, `ready_diagnostics`, `active_infusions`, and `scenario_difficulty`. Diagnostics are delayed rather than instantaneous: labs must be ordered, simulated time must pass, and the resulting study must then be retrieved from `ready_diagnostics`.

The consumer-facing action space exposes 17 tools across five clinical categories, while the underlying runtime supports a broader internal catalog for richer scenarios and evaluation.

- Assessment: `get_vitals`, `check_deterioration`, `summarize_state`
- Airway/breathing: `give_oxygen`, `airway_support`, `needle_decompression`
- Circulation: `control_bleeding`, `give_fluids`, `give_pressor`
- Diagnostics: `get_blood_gas`, `get_cbc`, `get_bmp`
- Procedure/time: `perform_pericardiocentesis`, `advance_time`

Internally, the runtime exposes a 68-entry public catalog, 64 of which are clinically usable engine-backed tools. Four actions remain explicitly unsupported in the local Pulse build because the corresponding substance files are not available: atropine, dopamine, plasma, and massive transfusion protocol. Those calls fail cleanly with structured `UNSUPPORTED_BY_ENGINE` responses rather than crashing the episode.

The reward function matters because it is the mechanism that turns physiology into protocol learning rather than reward hacking.

```text
R_t = 0.35 × MAP_stability
    + 0.25 × SpO2_efficiency  
    + 0.20 × lactate_trend
    + 0.10 × intervention_safety
    + 0.10 × diagnostic_timeliness
    + R_terminal (on episode end)
```

`MAP_stability` rewards restoration of perfusion pressure, while `SpO2_efficiency` rewards oxygenation improvement without divorcing it from overall trajectory. `lactate_trend` keeps the signal tied to shock reversal rather than superficial vital-sign normalization.

`intervention_safety` is where the environment enforces correct order. Hard penalties are applied for wrong clinical sequencing, including fluids before decompression (`-0.8`), pressors before adequate volume (`-0.5`), and succinylcholine without a secured airway path (`-1.0`). `diagnostic_timeliness` rewards early information gathering and retrieval of delayed studies when they become available.

The terminal term includes survival bonus, time efficiency, milestone sequence quality, and a difficulty multiplier so the agent is rewarded not only for survival but for surviving the right way. Anti-exploitation guards also penalize repeated tool spam, ignored ready diagnostics, and wasteful late-stage action loops.

The time pressure mechanic makes hesitation clinically meaningful. After three minutes of simulated time without stabilization, a deterioration multiplier activates at `1.0` and increases by `0.15` per minute per severity unit, while intervention effectiveness begins to decay. This reflects the reality of the golden hour: delayed treatment is not just later treatment, it is weaker treatment.

## Patient profiles

The patient corpus is a measured result, not a cosmetic roster. Twenty baseline Pulse patient profiles were run through a standardized trauma challenge protocol and ranked by observed physiological resilience across post-insult MAP, SpO2, shock index, mental status, and a short no-intervention survival window. That ranking produced three tiers that are intentionally data-driven rather than name-driven.

| Tier | Patients | Characteristics |
|---|---|---|
| Easy (7) | Bradycardic, Nathan, StandardMale, DefaultMale, Overweight, Carol, Jeff | Higher baseline cardiovascular reserve, tolerated standardized trauma challenge |
| Medium (7) | Jane, Cynthia, Underweight, DefaultFemale, Rick, Soldier, ExtremeMale | Moderate resilience, meaningful intervention required |
| Hard (6) | StandardFemale, Joel, Tachycardic, ExtremeFemale, Gus, Hassan | Most fragile under trauma insult, smallest intervention window |

Several assignments are counterintuitive on purpose. Bradycardic landed in the easy tier and StandardFemale landed in the hard tier because the ranking came from measured physiology under the same challenge, not from the surface semantics of the patient names.

## The three golden scenarios

### Scenario 1: Class III hemorrhagic shock

This scenario matters because it teaches the most basic circulation lesson in trauma care: pressure without volume is not resuscitation. The injuries are a single-compartment hemorrhage at `150 mL/min`, and the correct path is tourniquet, then crystalloid, then norepinephrine. The teaching point is volume before pressors, and the survival window is approximately eight simulated minutes.

### Scenario 2: Tension pneumothorax masquerading as shock (DEMO SCENARIO)

This is the demo case because it cleanly separates protocol learning from pattern matching. The patient presents with abdominal hemorrhage at `80 mL/min` plus a left tension pneumothorax, creating a trap in which shock-like hypotension tempts the wrong intervention. The correct path is auscultation, POCUS, needle decompression, then crystalloid, then norepinephrine; the teaching point is **diagnose before treating**. This is the case where **naive agent dies, trained agent survives**: a naive policy gives fluids into unresolved obstructive physiology and loses the patient, while the trained path decompresses first and SpO2 jumps from `0.84` to `0.99` in real Pulse physiology. The survival window is about six simulated minutes.

### Scenario 3: Cardiac tamponade after penetrating chest trauma

This scenario matters because it shows that some shock states cannot be solved pharmacologically. The injuries are pericardial effusion at severity `0.7` plus thoracic hemorrhage, producing Beck’s-triad-like obstructive physiology in which fluid resuscitation is only minimally effective until the obstruction is relieved. The correct path is POCUS cardiac, then pericardiocentesis, then crystalloid; the teaching point is that obstructive shock requires mechanical relief. The survival window is roughly five simulated minutes.

## Adversarial evaluation system

The adversarial system turns the environment from a scenario suite into a robustness benchmark. For each of the 20 patients, the injury-stacking adversary runs a fixed combo ladder of increasing simultaneous injury burden and records the first combination the policy cannot survive.

1. `tension_pneumothorax`
2. `hemorrhagic_shock`
3. `cardiac_tamponade`
4. `tension_pneumothorax + hemorrhagic_shock`
5. `hemorrhagic_shock + cardiac_tamponade`
6. `tension_pneumothorax + hemorrhagic_shock + cardiac_tamponade`

The results are already informative as a research finding. Across all 20 patients and all 6 combos, `120/120` generated resets succeeded. Under the expert policy, `7/20` patients survived `hemorrhagic_shock + cardiac_tamponade` at severity `0.7`, while `0/20` survived the triple-threat combination at the same severity. Each patient’s failure threshold is recorded with both `breaking_combo and breaking_severity`, and if a combo is terminal at reset the system automatically backs severity down in `0.1` steps to find an evaluable threshold instead of poisoning the map with reset deaths.

Hassan is a good example of why this matters clinically. That patient survived all three single-injury cases and also survived the pneumo-plus-hemorrhage double, but failed on hemorrhage plus tamponade. The reason is medically meaningful: active blood loss and obstructive shock create a treatment conflict with no clean linear ATLS sequence, so the benchmark reveals a real decision bottleneck rather than a synthetic “hard mode.”

## ATLS judge

The ATLS judge matters because judges and collaborators need a human-readable protocol score in addition to a scalar reward. Every observation includes an ATLS score computed by `atls_judge.py`, which evaluates action history together with patient state progression and produces a `0–100` score with pass, warn, and fail checks.

```text
ATLS Score: 96/100 — Textbook ATLS protocol
✓ PASS  Assessed before treating
✓ PASS  Decompressed before fluids
✓ PASS  Hemorrhage controlled early
✓ PASS  Labs ordered timely

ATLS Score: 14/100 — Critical protocol failure
✗ FAIL  Assessed before treating
✗ FAIL  Decompressed before fluids
✗ FAIL  Hemorrhage controlled early
✓ PASS  No dangerous drug interactions
```

The judge is intentionally heuristic rather than “oracle perfect,” but it does capture meaningful protocol structure. CPR is scored as valid when arrest appears in the patient state history, not only when arrest was manually induced, which allows the judge to handle true physiological deterioration rather than only scripted authoring events.

## PathologyArchitect

The PathologyArchitect matters because it turns the environment into a scenario generator rather than a fixed set of handcrafted cases. It takes `(patient_id, injury_type, severity)` and returns a valid scenario blueprint consumable by the environment. Two HTTP endpoints are exposed for this workflow: `GET /pathology/library` and `POST /pathology/generate`. Supported generated injury types are `tension_pneumothorax`, `hemorrhagic_shock`, `cardiac_tamponade`, and `polytrauma`.

## Training

The training story matters because the environment is intended to teach policy, not only to host scripted demos.

```bash
hf jobs run \
  --with trl \
  --flavor t4-small \
  --env PULSE_ENV_URL=https://your-space.hf.space \
  -- python train_grpo.py
```

Pulse-ER trains with GRPO through TRL, using Qwen2.5-3B-Instruct with LoRA rank `16` for submission-facing runs and the mock backend for fast iteration before promotion to the real Pulse-backed evaluation path. The same reward formula above is used during training, which means protocol sequencing pressure is present during optimization and not added later as a hand-authored judge-only score. In verified policy ordering, the expert policy remains positive across scenarios, `llm_demo` remains positive on easy cases but degrades on hard ones, and both `random` and `no_action` reach `patient_death` on `3/4` real scenarios.

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

The architecture matters because clean separation is what lets the same environment support mock training, real Pulse evaluation, judge-facing explanation, and adversarial testing without contract drift. `pulse_engine_adapter.py` owns all Pulse interaction, state synthesis, and semantic operations. `tools.py` defines the tool registry and handlers, `reward_engine.py` owns dense and terminal scoring, `atls_judge.py` produces human-readable protocol evaluation, `patient_monitor.py` builds monitor payloads, `pathology_architect.py` generates new scenarios, `scenarios.py` contains patient pools and scenario registry, `injury_stack_adversary.py` runs robustness evaluation, `adapters.py` implements the mock backend with the full 17-tool contract, `app.py` exposes reset/step/health/pathology endpoints, and `train_grpo.py` is the GRPO training entrypoint.

## Research findings

The following results were produced by running the environment against the 20-patient corpus with the standardized trauma protocol. Policy separation is visible both qualitatively and numerically: on `hemorrhagic_shock`, the expert policy scored `8.33`, while `random` scored `-17.15` and `no_action` scored `-17.10`, both terminating in patient death. Difficulty calibration is also measurable rather than intuitive: the hard patients stabilized from post-insult MAP values in the `41–59` range with SpO2 in the `0.62–0.83` range, while easy patients remained closer to MAP in the `90s` and SpO2 around `0.95–0.96` under the same standardized challenge. The adversarial system showed the same separation at the population level, with `7/20` patients surviving the double-threat hemorrhage-plus-tamponade case and `0/20` surviving the triple-threat benchmark at severity `0.7`. Reward quality is also sequence-sensitive on matched seeds: the naive pneumothorax sequence scored `-0.838`, while the correct decompression-first sequence scored `-0.068` on the same patient and seed.

## Limitations and future work

This environment is strongest when it is precise about what is solved and what remains open.

- Four tools remain unsupported in the local Pulse 4.3.2 build because the required substance files are unavailable: atropine, dopamine, plasma, and MTP. These return structured `UNSUPPORTED_BY_ENGINE`.
- `position_patient` is currently context-only because this local build does not expose a native Pulse position action.
- The triple-threat combination is universally lethal at severity `0.7` for the current trained agent and therefore functions as an unsolved benchmark level rather than a solved capability.

These limitations point directly to the next research steps rather than to hand-waving roadmap promises.

- Add a severity-escalation adversary on top of injury stacking so each patient can be benchmarked with binary-searched failure thresholds.
- Extend beyond the golden hour into ventilator weaning and prolonged-care scenarios.
- Introduce multi-injury complication events grounded in validated physiology, such as rebound pneumothorax and transfusion reactions.
- Run larger-model training with the full 64-tool clinical surface exposed to the learner.
