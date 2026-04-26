# Person 1 Integration Handoff

This note is for the real Pulse runtime integration workstream.

## Goal

Replace the mock runtime path with a real Pulse-backed runtime path without breaking Person 2's consumer-side code.

The intended swap is:

- current: `MockPulseAdapter`
- target: `RealPulseAdapter`

The swap should not require rewriting:

- `models.py`
- `client.py`
- `rewards.py`
- `smoke_test.py`
- `eval_mock.py`

## Files To Read First

- [SPEC.md](./SPEC.md)
- [models.py](./models.py)
- [server/adapters.py](./server/adapters.py)
- [server/scenarios.py](./server/scenarios.py)
- [README.md](./README.md)

## What Person 1 Owns

- real Pulse engine connection
- real patient state extraction
- real tool execution
- real scenario progression
- `RealPulseAdapter`

## What Must Stay Stable

Do not change these without explicitly syncing first:

- field names
- units
- tool names
- `mental_status` enum values
- top-level response envelope

## Required Public State

The real adapter must populate the same public contract fields:

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

Notes:

- `blood_volume_ml` may be `null` temporarily if unavailable, but the field must still exist.
- `mental_status` must use:
  - `alert`
  - `verbal`
  - `pain`
  - `unresponsive`

## Required Tool Names

The initial tool set is frozen as:

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

## Expected Adapter Interface

The real adapter should implement the same interface shape as `PatientBackend`:

```python
class PatientBackend(ABC):
    def reset(self, scenario_id: str | None = None) -> EnvironmentResponse: ...
    def step(self, action: ToolAction) -> EnvironmentResponse: ...
    def get_state(self) -> PatientState: ...
```

## Expected Step Envelope

Every step should return:

- `observation`
- `reward`
- `done`
- `metadata`
- `tool_result`
- `error`

Errors must be structured, not raw exceptions or plain strings.

## Recommended Implementation Path

1. Add `RealPulseAdapter` to [server/adapters.py](./server/adapters.py) or a nearby runtime-specific module.
2. Keep `MockPulseAdapter` untouched so Person 2 regression checks continue to work.
3. Use the real runtime to populate `PatientState`.
4. Preserve the exact tool names and response envelope.
5. Only after the real adapter matches the contract should it be wired into the OpenEnv environment path.

## Integration Acceptance Checklist

Before calling the integration ready, confirm all of the following:

- same top-level field names as `SPEC.md`
- same units as `SPEC.md`
- same tool names
- invalid tool calls return structured errors
- mock-side scripts still work without model rewrites
- replacing `MockPulseAdapter` with `RealPulseAdapter` does not require consumer-side contract changes

## Fast Verification Steps

Once the real adapter exists, Person 1 and Person 2 should jointly verify:

1. `reset()` returns a full `PatientState`
2. `step()` returns the same envelope shape as the mock
3. `blood_volume_ml` is populated or intentionally `null`
4. good interventions improve outcomes relative to bad or no-action paths
5. no contract drift exists between mock and real outputs

## Most Important Rule

If the real backend needs a contract change, do not silently edit the consumer-side models first.

Sync on the change, update `SPEC.md`, then update both sides together.
