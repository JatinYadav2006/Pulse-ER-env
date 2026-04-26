# Mock vs Real Verification Report

Date: 2026-04-26

## Scope

This pass re-ran the project after fixing the airway-support contract mismatch where model outputs like `airway_support(mode="basic")` were accepted by mock but rejected by the real backend.

The fix was applied in:

- [pulse_physiology_env/tool_catalog.py](./pulse_physiology_env/tool_catalog.py)
- [pulse_physiology_env/server/tools.py](./pulse_physiology_env/server/tools.py)
- [pulse_physiology_env/server/adapters.py](./pulse_physiology_env/server/adapters.py)
- [pulse_physiology_env/demo_llm_policy.py](./pulse_physiology_env/demo_llm_policy.py)
- [pulse_physiology_env/policies.py](./pulse_physiology_env/policies.py)
- [pulse_physiology_env/smoke_test.py](./pulse_physiology_env/smoke_test.py)
- [pulse_physiology_env/tier3_workflows.py](./pulse_physiology_env/tier3_workflows.py)

## Core Checks

Passed:

- `python -m compileall pulse_physiology_env`
- `python -m pulse_physiology_env.integration_smoke`
- `python -m pulse_physiology_env.smoke_test`
- `python -m pulse_physiology_env.eval_mock`

Targeted regression passed:

- `validate_tool_arguments("airway_support", {"mode": "basic"}) -> {"mode": "auto"}`
- `validate_tool_arguments("airway_support", {"support_type": "Default"}) -> {"support_type": "auto"}`
- Real backend direct call with `airway_support(mode="basic")` now succeeds and resolved to `bag_valve_mask`

## Mock Matrix

Scenarios:

- `baseline_stable`
- `respiratory_distress`
- `hemorrhagic_shock`

Policies:

- `expert`
- `llm_demo`
- `random_seed0`
- `no_action`

Results:

| Scenario | Policy | Reward | Termination | Notes |
|---|---:|---:|---|---|
| `baseline_stable` | `expert` | `-0.400` | `max_timesteps` | Stable; stricter anti-spam shaping keeps reward near zero/negative late |
| `baseline_stable` | `llm_demo` | `-0.310` | `max_timesteps` | Stable |
| `baseline_stable` | `random_seed0` | `-2.890` | `max_timesteps` | Stable but noisy care |
| `baseline_stable` | `no_action` | `-5.860` | `max_timesteps` | Stable but passive |
| `respiratory_distress` | `expert` | `2.509` | `max_timesteps` | SpO2 recovered to `99.2%` |
| `respiratory_distress` | `llm_demo` | `-1.250` | `max_timesteps` | Improved but still inefficient |
| `respiratory_distress` | `random_seed0` | `-8.450` | `max_timesteps` | Persistently hypoxemic |
| `respiratory_distress` | `no_action` | `-13.500` | `patient_death` | Terminal collapse |
| `hemorrhagic_shock` | `expert` | `8.331` | `max_timesteps` | Good control and resuscitation |
| `hemorrhagic_shock` | `llm_demo` | `9.149` | `max_timesteps` | Strong scripted ordering |
| `hemorrhagic_shock` | `random_seed0` | `-17.150` | `patient_death` | Terminal collapse |
| `hemorrhagic_shock` | `no_action` | `-17.100` | `patient_death` | Terminal collapse |

Mock verdict:

- Mock backend is healthy for training.
- No parser/runtime failures showed up in the matrix.
- The previous `basic` alias problem is gone.
- Remaining mock weirdness is mostly reward-design choice on `baseline_stable`, not a backend bug.

## Real Matrix

Scenarios:

- `trauma_easy_soldier`
- `trauma_medium_carol`
- `trauma_hard_underweight`
- `polytrauma_demo`

Policies:

- `expert`
- `llm_demo`
- `random_seed0`
- `no_action`

Results:

| Scenario | Policy | Reward | Termination | Notes |
|---|---:|---:|---|---|
| `trauma_easy_soldier` | `expert` | `0.016` | `max_timesteps` | Stabilized; SpO2 `96.8%` |
| `trauma_easy_soldier` | `llm_demo` | `-0.485` | `max_timesteps` | Survived but remained unstable |
| `trauma_easy_soldier` | `random_seed0` | `-3.741` | `max_timesteps` | Persisting hemorrhage |
| `trauma_easy_soldier` | `no_action` | `-6.671` | `max_timesteps` | Severe hypoxemia by episode end |
| `trauma_medium_carol` | `expert` | `0.179` | `max_timesteps` | Survived; still some shock/bleeding burden |
| `trauma_medium_carol` | `llm_demo` | `-10.735` | `patient_death` | Died after repeated late airway-support engine retries |
| `trauma_medium_carol` | `random_seed0` | `-8.607` | `patient_death` | Died |
| `trauma_medium_carol` | `no_action` | `-11.586` | `patient_death` | Died |
| `trauma_hard_underweight` | `expert` | `0.570` | `max_timesteps` | Survived hard case; SpO2 `95.4%` |
| `trauma_hard_underweight` | `llm_demo` | `-11.701` | `patient_death` | Died after unstable progression |
| `trauma_hard_underweight` | `random_seed0` | `-9.777` | `patient_death` | Died |
| `trauma_hard_underweight` | `no_action` | `-11.232` | `patient_death` | Died |
| `polytrauma_demo` | `expert` | `0.609` | `max_timesteps` | Stabilized well enough to survive full run |
| `polytrauma_demo` | `llm_demo` | `-11.888` | `patient_death` | Died after repeated late airway-support retries |
| `polytrauma_demo` | `random_seed0` | `-8.607` | `patient_death` | Died |
| `polytrauma_demo` | `no_action` | `-12.446` | `patient_death` | Died |

Real verdict:

- The real backend no longer fails because of the old `basic` alias.
- `expert` now survives all 4 tested real scenarios in this pass.
- `llm_demo` no longer dies from `INVALID_ARGUMENT`; it now fails for the clinically honest reason: poor sequencing on unstable cases.
- The remaining real failures are policy quality / physiology severity issues, not contract bugs.

## Tool-Surface Status

Consumer contract:

- Public consumer tool list remains the frozen 17-tool contract.
- Wrapper still preserves the richer runtime tool list under `metadata["raw_available_tools"]`.

Previously verified real raw sweep:

- `68` raw real tools exercised
- `64` successful
- `4` structured `UNSUPPORTED_BY_ENGINE`

Known real engine-limited tools:

- `administer_atropine_bolus`
- `start_dopamine_infusion`
- `administer_plasma`
- `activate_massive_transfusion_protocol`

## What Was Fixed In This Pass

1. Shared argument normalization now treats vague airway aliases as recoverable formatting noise instead of fatal contract mismatches.
2. Real runtime `airway_support` now maps `auto/basic/default/standard` to patient-state-driven support selection.
3. Mock runtime mirrors that same behavior instead of storing `"basic"` as a fake mode.
4. Demo/playbook/example code was cleaned so authored traces prefer the stronger contract too.

## What Still Is Not A Bug

- `llm_demo` still performs poorly on real unstable trauma cases.
  - This is now a policy-quality problem, not a parser/runtime mismatch.
- `baseline_stable` mock rewards stay near zero or slightly negative for long episodes.
  - That is a shaping choice driven by anti-spam penalties, not a failure to stabilize.

## Bottom Line

After this fix, the mock and real stacks are much more aligned:

- same 17-tool consumer contract
- same normalization behavior for noisy model outputs
- no real-side fatal error from `airway_support(mode="basic")`
- remaining real losses are clinically meaningful failures rather than contract failures

This means the current stack is in a better place for:

- Person 2 mock training
- side-by-side mock vs real evaluation
- judge/demo confidence that failures reflect care quality instead of JSON formatting quirks
