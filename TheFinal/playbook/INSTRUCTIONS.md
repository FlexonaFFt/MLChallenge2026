# Instructions

## Agent Rules

- First read `README.md`, `POLICY.md`, `TASKS.md`, and the current workspace README.
- Do not solve by guessing. Inspect files, metric, target, sample submission, and data shape.
- Start with the simplest valid baseline.
- Change one major thing per experiment.
- Record every experiment before starting the next one.
- Preserve working code and submissions.
- Do not chase complex ideas until a valid submission exists.

## Four-Hour Flow

### T+00:00 - T+00:20

- Read task statement.
- Identify input, output, metric, submission format.
- Inspect `train`, `test`, and `sample_submission`.
- Create a first validation split.
- Write task facts to `workspaces/<task_name>/README.md`.

### T+00:20 - T+01:00

- Build the first valid baseline.
- Produce a valid submission file.
- Run submission sanity checks.
- Log baseline as `e001`.

### T+01:00 - T+02:30

- Improve features/model.
- Compare every change on the same validation.
- Use local LLMs only when they create measurable signal or speed up analysis.

### T+02:30 - T+03:30

- Ensemble or refine the best candidates.
- Check leakage and overfitting.
- Freeze the safest submission candidate.

### T+03:30 - T+04:00

- Stop risky experiments.
- Re-run final inference if needed.
- Validate file format.
- Write final notes to `SUBMISSION.md`.

## Experiment Format

Every experiment gets an id:

```text
e001, e002, e003
```

Each experiment must record:

- idea
- files changed
- command
- validation score
- public score if known
- status
- next action

## Handoff Rule

When local brainstorming is useful, write the result to:

```text
workspaces/<task_name>/HANDOFF_TO_SRC_CODE.md
```

The handoff must be short enough for `src code` to execute without re-reading all brainstorm notes.

