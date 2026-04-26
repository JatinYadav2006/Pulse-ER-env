# Pulse-ER Demo Script

This is the judge-facing walkthrough for the current hackathon build.

The goal is to show three things clearly:

1. Pulse-ER is a tool-driven clinical environment, not just a vitals dashboard.
2. Good decisions produce better physiological outcomes than bad or delayed care.
3. The architecture is ready to swap from deterministic mock episodes to the real Pulse runtime.

## Demo Story

Use this one-liner up front:

> Pulse-ER turns physiology into an interactive clinical environment where an agent can assess a patient, choose interventions, observe the response over time, and be scored on the quality of care.

Then anchor the structure:

- Tier 1: read and assess
- Tier 2: intervene
- Tier 3: recommend and orchestrate next steps

## Fast Demo Order

Run the demo in this order:

1. Show the frozen contract
2. Show the benchmark harness
3. Show one strong respiratory episode
4. Show one strong hemorrhagic shock episode
5. Close on the mock-to-real swap story

This order keeps the demo focused on product value, not implementation noise.

## Suggested 3-4 Minute Script

### 1. Open with the problem

Say:

> In emergency care, the hard part is not just reading vitals. It is deciding the next best intervention under time pressure and seeing whether the patient stabilizes or deteriorates. Pulse-ER gives an agent that kind of interactive clinical environment.

### 2. Show the contract

Open [SPEC.md](./SPEC.md).

Say:

> We froze a shared contract first so the real physiology runtime and the training pipeline could be built in parallel without integration drift.

Call out only these fields:

- `heart_rate_bpm`
- `systolic_bp_mmhg`
- `diastolic_bp_mmhg`
- `spo2`
- `respiration_rate_bpm`
- `blood_volume_ml`
- `mental_status`
- `active_alerts`
- `done`

### 3. Show the benchmark

Run:

```powershell
.\.venv311\Scripts\python.exe -m pulse_physiology_env.eval_mock
```

Say:

> We benchmark four policy levels: expert scripted care, an LLM-style tool policy, random actions, and no-action. The ranking we want is expert above LLM, and both far above random and no-action.

Current expected story:

- expert average: `4.613`
- llm_demo average: `3.930`
- random average: `-5.829`
- no_action average: `-9.053`

### 4. Show respiratory distress

Run:

```powershell
.\.venv311\Scripts\python.exe -m pulse_physiology_env.run_mock_episode --scenario respiratory_distress --policy expert --max-steps 8
```

Say:

> In respiratory distress, the gold path is oxygen, positioning, airway support, then reassessment over time. The patient improves and clears active alerts.

What judges should notice:

- SpO2 rises
- respiratory rate falls
- alerts disappear

### 5. Show hemorrhagic shock

Run:

```powershell
.\.venv311\Scripts\python.exe -m pulse_physiology_env.demo_llm_policy --scenario hemorrhagic_shock --max-steps 8
```

Say:

> In hemorrhagic shock, the agent identifies blood loss, controls bleeding, gives fluids, adds oxygen and positioning support, and observes stabilization.

What judges should notice:

- blood loss is treated as the primary driver
- fluids improve perfusion
- oxygen and positioning reduce compensatory stress
- the patient becomes more stable instead of spiraling

### 6. Close on architecture

Open [README.md](./README.md) or [PERSON1_HANDOFF.md](./PERSON1_HANDOFF.md).

Say:

> This mock path is not throwaway work. The contract, parser, policies, reward harness, and trajectory generation are already built around a stable interface. The next step is swapping in the real Pulse-backed adapter without rewriting the consumer side.

## Exact Commands

Use these commands from the repo root:

```powershell
.\.venv311\Scripts\python.exe -m pulse_physiology_env.smoke_test
.\.venv311\Scripts\python.exe -m pulse_physiology_env.eval_mock
.\.venv311\Scripts\python.exe -m pulse_physiology_env.run_mock_episode --scenario respiratory_distress --policy expert --max-steps 8
.\.venv311\Scripts\python.exe -m pulse_physiology_env.demo_llm_policy --scenario hemorrhagic_shock --max-steps 8
```

## What Not To Over-Explain

Avoid spending judge time on:

- detailed class structures
- every mock file
- raw reward math
- internal parser implementation

Keep the conversation on:

- clinical reasoning loop
- visible patient response
- benchmark separation
- mock-to-real architecture

## Backup Plan

If live runtime output is noisy, fall back to:

- [seed_trajectories_summary.json](./artifacts/seed_trajectories_summary.json)
- [SPEC.md](./SPEC.md)
- [README.md](./README.md)

Use this line:

> Even offline, the benchmark artifacts show the same core story: expert care outperforms LLM heuristics, and both outperform random or delayed care by a wide margin.

## Demo Success Criteria

The demo is successful if a judge leaves with these conclusions:

- this is a clinical decision environment, not a toy simulator
- interventions visibly affect physiology
- the benchmark is disciplined
- the architecture is ready for real Pulse runtime integration
