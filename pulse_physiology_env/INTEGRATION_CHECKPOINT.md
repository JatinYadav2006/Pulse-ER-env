# Real Integration Checkpoint

This is the operational checkpoint for Person 1 and Person 2 before calling the real Pulse integration ready.

It is stricter than the general handoff note in [PERSON1_HANDOFF.md](./PERSON1_HANDOFF.md).

## Goal

Confirm that the real Pulse-backed runtime can replace the mock backend boundary without breaking:

- contract stability
- tool semantics
- benchmark harnesses
- demo flow

## Read First

- [SPEC.md](./SPEC.md)
- [PERSON1_HANDOFF.md](./PERSON1_HANDOFF.md)
- [README.md](./README.md)
- [server/adapters.py](./server/adapters.py)
- [models.py](./models.py)

## Required Output Shape

The real runtime must return the same top-level envelope as the mock path:

- `observation`
- `reward`
- `done`
- `metadata`
- `tool_result`
- `error`

And the public state must preserve:

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

## Tool Contract Check

These tool names are frozen:

- `get_vitals`
- `advance_time`
- `give_oxygen`
- `give_fluids`
- `control_bleeding`
- `position_patient`
- `airway_support`
- `summarize_state`
- `check_deterioration`
- `recommend_next_step`

For each tool, verify:

- it accepts the same argument names
- invalid arguments return structured errors
- `tool_result` is always populated for handled calls
- `changed_fields` is clinically plausible

## Integration Checklist

Mark all of these before merging the real adapter into the main demo path:

- [ ] `reset()` returns a complete observation envelope
- [ ] `step()` returns the same envelope shape as the mock path
- [ ] `done == observation.done`
- [ ] `mental_status` uses the frozen enum values only
- [ ] `blood_volume_ml` exists in schema even if temporarily `null`
- [ ] unknown tools return structured `UNKNOWN_TOOL`
- [ ] bad arguments return structured `INVALID_ARGUMENT`
- [ ] `metadata.available_tools` reflects the actually usable tools
- [ ] no consumer-side file needed a contract rewrite

## Side-By-Side Verification

Run the same logical checks on both backends.

### Contract sanity

- confirm field names
- confirm units
- confirm enum values
- confirm top-level keys

### Behavior sanity

- good respiratory interventions improve oxygenation
- bad or delayed respiratory care worsens state
- hemorrhage control plus fluids improves perfusion relative to no-action
- unsupported actions fail cleanly instead of crashing

## Required Joint Test Session

Person 1 and Person 2 should do this together.

### Session 1: Envelope validation

Verify:

- reset shape
- step shape
- tool_result shape
- error shape

### Session 2: Clinical sanity

Run one respiratory case and one hemorrhage case.

Verify:

- the intervention sequence is clinically recognizable
- physiology moves in the expected direction
- the real runtime does not silently omit required fields

### Session 3: Consumer-side compatibility

Verify that Person 2 code still runs with minimal or no contract changes in:

- [client.py](./client.py)
- [prompt_builder.py](./prompt_builder.py)
- [tool_parser.py](./tool_parser.py)
- [policies.py](./policies.py)
- [episode_runner.py](./episode_runner.py)

## Current Mock Benchmark Reference

Use these as the current mock-side reference numbers:

- expert average: `4.613`
- llm_demo average: `3.930`
- random average: `-5.829`
- no_action average: `-9.053`

These are not required exact real-runtime values.

They are only a reference for the shape of separation we want:

- strong policy better than weak policy
- informed action better than delay
- interventions produce meaningful physiological change

## Merge Gate

Do not call the real integration complete until these are true:

- Person 1 can show real tool execution with the frozen contract
- Person 2 can point existing mock-side consumer code at the real path with minimal edits
- a judge-facing demo can run without explaining away contract drift

## Final Rule

If the real runtime needs a contract change, update:

1. [SPEC.md](./SPEC.md)
2. consumer-side models and parser logic
3. real runtime adapter

Do not change only one side and hope the mismatch is harmless.
