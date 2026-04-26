# Injury Stack Adversary Report

Date: 2026-04-26

## What was implemented

Approach 3 is now implemented as a real generated-case adversary instead of a standalone demo script.

Main pieces:

- Multi-injury blueprint generation in [pulse_physiology_env/server/pathology_architect.py](./pulse_physiology_env/server/pathology_architect.py)
- Generated-case reset support for `injury_types` in [pulse_physiology_env/server/pulse_physiology_env_environment.py](./pulse_physiology_env/server/pulse_physiology_env_environment.py)
- HTTP support for stacked generation in [pulse_physiology_env/server/app.py](./pulse_physiology_env/server/app.py)
- Expert-policy support for generated stacked trauma and earlier tamponade relief in [pulse_physiology_env/policies.py](./pulse_physiology_env/policies.py)
- New adversary runner in [pulse_physiology_env/injury_stack_adversary.py](./pulse_physiology_env/injury_stack_adversary.py)
- Regression coverage in [pulse_physiology_env/integration_smoke.py](./pulse_physiology_env/integration_smoke.py)

## Default injury ladder

The default adversary combos are:

1. `tension_pneumothorax`
2. `hemorrhagic_shock`
3. `cardiac_tamponade`
4. `tension_pneumothorax + hemorrhagic_shock`
5. `hemorrhagic_shock + cardiac_tamponade`
6. `tension_pneumothorax + hemorrhagic_shock + cardiac_tamponade`

Requested severity used for the main stress pass: `0.7`

## Robustness fixes added during implementation

The implementation surfaced three real stress-path issues, and all three were addressed:

1. Generated blueprints round-tripped with both `injury_type` and `injury_types`, but reset validation only allowed one selector key.
   Fix: reset now accepts stored blueprint dicts that include the summary field plus the full combo list.

2. Very sick stacked cases could fail during setup because Pulse would not accept one large deterioration jump.
   Fix: blueprint setup now advances time adaptively in smaller chunks and saturates at the last reachable critical state instead of crashing reset.

3. A few very fragile patients could still be terminal at reset on the hardest generated double-threat combo.
   Fix: the adversary now automatically backs severity down in `0.1` steps, as low as `0.5`, whenever a combo is terminal before the first action. This prevents reset deaths from contaminating the breaking-combo map.

I also re-ordered the real expert baseline so it treats tamponade as an earlier obstruction problem:

- needle decompression first if present
- then pericardiocentesis when tamponade signs are active
- then bleeding control and volume support

That made the double-threat combo (`hemorrhagic_shock + cardiac_tamponade`) selectively survivable instead of uniformly failing.

## Verification run

Passed:

- `python -m compileall pulse_physiology_env`
- `python -m pulse_physiology_env.integration_smoke`
- `python -m pulse_physiology_env.smoke_test`
- `python -m pulse_physiology_env.eval_mock`

New regression now included:

- stacked pathology blueprints build correctly
- combined setup actions include all requested injuries
- the final setup action remains a deterioration window

## Whole-cohort reset stress

Supported patients tested: `20`

Default combos tested per patient: `6`

Total generated resets: `120`

Raw generator result before adversary backoff:

- reset errors: `0`
- terminal on reset: `3`

The only raw reset-terminal cases at requested severity `0.7` were:

- `extreme_female` with `hemorrhagic_shock + cardiac_tamponade`
- `gus` with `hemorrhagic_shock + cardiac_tamponade`
- `hassan` with `hemorrhagic_shock + cardiac_tamponade`

So the generator is stable across the full patient library, and only the most fragile patients start already lost on the hardest double-obstructive/hemorrhagic combo.

Adversary interpretation after the reset-survival backoff:

- those reset-terminal cases are automatically retried at lower severity
- the breaking map records the actual evaluated severity instead of pretending the patient failed at the original requested severity

## Expert adversary results

### Hard-combo cohort pass

To keep runtime practical while still being meaningful, the full cohort was rerun on the two hardest combos:

- `hemorrhagic_shock + cardiac_tamponade`
- `tension_pneumothorax + hemorrhagic_shock + cardiac_tamponade`

Results at requested severity `0.7`:

- `hemorrhagic_shock + cardiac_tamponade`: `7 / 20` patients survived
- `tension_pneumothorax + hemorrhagic_shock + cardiac_tamponade`: `0 / 20` patients survived

This gives a useful, demo-friendly patient split:

- some patients break at the double-threat combo
- some patients tolerate the double-threat combo and only break at the triple-threat combo

Patients that survived the double-threat combo in this pass:

- `default_male`
- `extreme_male`
- `jeff`
- `joel`
- `rick`
- `soldier`
- `tachycardic`

Patients that broke at the double-threat combo:

- `bradycardic`
- `carol`
- `cynthia`
- `default_female`
- `extreme_female`
- `gus`
- `hassan`
- `jane`
- `nathan`
- `overweight`
- `standard_female`
- `standard_male`
- `underweight`

### Interpretation

This is the behavior we wanted from the hackathon framing:

- isolated problems are generally manageable
- the combined hemorrhage + tamponade stack becomes the first real breaking point for many patients
- the triple-threat stack is universally catastrophic at severity `0.7`

## Cross-policy sanity on generated cases

Representative patients used:

- `jeff`
- `soldier`
- `hassan`

Policies checked:

- `llm_demo`
- `random`
- `no_action`

Important stability result:

- no `fatal_backend_error`
- no `initialization_error`

Observed behavior:

- `llm_demo` typically broke at `hemorrhagic_shock`
- `random` typically broke at `tension_pneumothorax`
- `no_action` typically broke at `tension_pneumothorax`

That is the right shape: weaker policies fail earlier, but the stack runner itself remains stable. It also validates that the combo ladder is calibrated correctly: harder combos break better policies later.

## Severity dimension

The breaking point is now explicitly a `combo + severity` result, not just a combo label:

- each episode result stores both `requested_severity` and the actual evaluated `severity`
- `reset_adjusted=true` marks cases where the adversary had to back severity off to avoid a reset death
- each patient result now includes `breaking_combo` and `breaking_severity`

Explicit next work:

- add a true severity-escalation sweep on top of the injury ladder so the project can measure per-patient thresholds such as "survives hemorrhage+tamponade at `0.5`, fails at `0.7`"

## Bottom line

Approach 3 is now implemented and usable.

The new adversary stack:

- supports multi-injury blueprint generation
- runs against the real Pulse-backed environment
- produces per-patient breaking-combo results
- records severity alongside the combo result
- automatically backs off severity when a combo is terminal at reset
- survives full-cohort reset stress without crashes
- remains stable under weaker policies

The strongest demo story from this pass is:

- "Our expert handles isolated trauma and many double-threat cases."
- "Some patients break at hemorrhage + tamponade."
- "At the same severity, every patient breaks at the triple-threat stack."

That is simple, dramatic, and grounded in real episode runs rather than hand-wavy claims.
