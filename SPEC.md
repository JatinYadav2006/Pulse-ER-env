# Pulse-ER Contract Spec

## Purpose

This file freezes the interface between:

- Person 1: Pulse engine/runtime implementation
- Person 2: training pipeline, tool orchestration, evaluation, and product layer

The goal is to let both people work in parallel with minimal integration risk.

## Rules

- Do not rename fields without explicitly notifying the other person.
- Do not change units after implementation starts.
- New optional fields may be added, but existing fields must remain stable.
- If a field is temporarily unavailable in the real backend, return `null` only if the schema allows it. Do not silently omit fields.
- Tool failures must return structured errors, not crashes or plain strings.

## Canonical Units

- Time: seconds
- Heart rate: beats per minute
- Blood pressure: mmHg
- SpO2: 0.0 to 1.0
- Respiration rate: breaths per minute
- Blood volume: mL

## Patient State Contract

This is the minimum shared state shape that both sides build against.

```json
{
  "scenario_id": "respiratory_distress",
  "patient_id": "standard_male",
  "sim_time_s": 120.0,
  "heart_rate_bpm": 104.0,
  "systolic_bp_mmhg": 108.0,
  "diastolic_bp_mmhg": 68.0,
  "spo2": 0.91,
  "respiration_rate_bpm": 28.0,
  "blood_volume_ml": 5200.0,
  "mental_status": "alert",
  "active_alerts": [
    "tachycardia",
    "hypoxemia"
  ],
  "done": false
}
```

## Patient State Fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `scenario_id` | `str` | yes | Scenario currently being simulated |
| `patient_id` | `str` | yes | Current patient template or identifier |
| `sim_time_s` | `float` | yes | Simulation time in seconds |
| `heart_rate_bpm` | `float` | yes | Heart rate |
| `systolic_bp_mmhg` | `float` | yes | Systolic blood pressure |
| `diastolic_bp_mmhg` | `float` | yes | Diastolic blood pressure |
| `spo2` | `float` | yes | Oxygen saturation from `0.0` to `1.0` |
| `respiration_rate_bpm` | `float` | yes | Respiratory rate |
| `blood_volume_ml` | `float` | yes | Estimated blood volume |
| `mental_status` | `str` | yes | One of `alert`, `verbal`, `pain`, `unresponsive` |
| `active_alerts` | `list[str]` | yes | High-signal warnings used by downstream logic |
| `done` | `bool` | yes | Whether the episode or scenario is over |

## Initial Tool Set

These are the first tools both sides should build around.

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

More tools can be added later without breaking this contract.

## Tool Action Contract

Every action from the consumer side to the environment follows this shape.

```json
{
  "tool_name": "give_oxygen",
  "arguments": {
    "flow_lpm": 15
  },
  "reasoning": "Patient is hypoxemic and in respiratory distress."
}
```

## Tool Action Fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `tool_name` | `str` | yes | Must match a supported tool exactly |
| `arguments` | `dict[str, any]` | yes | Tool parameters |
| `reasoning` | `str` | no | Optional human-readable rationale |

## Tool Result Contract

Every successful or failed tool call returns a structured result object.

```json
{
  "tool_name": "give_oxygen",
  "success": true,
  "message": "High-flow oxygen started.",
  "state_changed": true,
  "changed_fields": [
    "spo2",
    "respiration_rate_bpm"
  ]
}
```

## Tool Result Fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `tool_name` | `str` | yes | Echoes the invoked tool |
| `success` | `bool` | yes | Whether the tool executed cleanly |
| `message` | `str` | yes | Short summary of the outcome |
| `state_changed` | `bool` | yes | Whether patient state changed |
| `changed_fields` | `list[str]` | yes | Which top-level `PatientState` fields changed |

## Error Contract

Tool failures must use structured errors.

```json
{
  "code": "INVALID_ARGUMENT",
  "message": "flow_lpm must be between 1 and 30",
  "retryable": false
}
```

## Error Fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `code` | `str` | yes | Machine-readable code |
| `message` | `str` | yes | Human-readable explanation |
| `retryable` | `bool` | yes | Whether the caller may retry after adjustment |

## Environment Response Contract

Every environment step returns this top-level shape.

```json
{
  "observation": {
    "scenario_id": "respiratory_distress",
    "patient_id": "standard_male",
    "sim_time_s": 130.0,
    "heart_rate_bpm": 101.0,
    "systolic_bp_mmhg": 110.0,
    "diastolic_bp_mmhg": 70.0,
    "spo2": 0.94,
    "respiration_rate_bpm": 24.0,
    "blood_volume_ml": 5200.0,
    "mental_status": "alert",
    "active_alerts": [
      "tachycardia"
    ],
    "done": false
  },
  "reward": 1.2,
  "done": false,
  "metadata": {
    "step_count": 4,
    "available_tools": [
      "get_vitals",
      "advance_time",
      "give_oxygen",
      "give_fluids"
    ]
  },
  "tool_result": {
    "tool_name": "give_oxygen",
    "success": true,
    "message": "High-flow oxygen started.",
    "state_changed": true,
    "changed_fields": [
      "spo2",
      "respiration_rate_bpm"
    ]
  },
  "error": null
}
```

## Response Field Rules

- `observation` always returns the latest full patient state.
- `reward` is always numeric.
- `done` must match `observation.done`.
- `metadata` may grow over time, but existing keys should not be removed casually.
- `tool_result` is present for a handled action, even if `success` is `false`.
- `error` is `null` on success.

## Required Integration Test

Before mock and real integration is considered complete, both sides must pass these checks:

- Same top-level field names
- Same units
- Same tool names
- Same response envelope
- Invalid tool calls return structured errors
- Good intervention paths improve reward relative to bad or no-action paths

## Swap Guarantee

Person 2 should be able to replace:

- `MockPulseAdapter`

with:

- `RealPulseAdapter`

without rewriting the policy, parser, reward harness, or consumer-side models.
