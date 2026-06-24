# Final ML Playbook

This folder is the local offline guide for a 4-hour ML competition.

## Agent Entry Point

Always start here.

Read in this order:

1. `POLICY.md`
2. `INSTRUCTIONS.md`
3. `TASKS.md`
4. `LLM_MODELS.md`
5. `workspaces/<current_task>/README.md`

During work:

- Keep notes in `workspaces/<current_task>/`.
- Use `T+MM` or `T+HH:MM` timestamps, not dates.
- Log every meaningful action in `LOG.md`.
- Log every tested idea in `EXPERIMENTS.md`.
- Log failures and fixes in `ERRORS.md`.
- Log submitted files in `SUBMISSION.md`.
- Keep `HANDOFF_TO_SRC_CODE.md` ready for the VM agent.

## Competition Workflow

1. Copy `workspaces/_template` to `workspaces/<task_name>`.
2. Fill `workspaces/<task_name>/README.md` from the task statement.
3. Use `TASKS.md` to classify the task and pick the first baseline.
4. Use `LLM_MODELS.md` only when a local model helps the metric.
5. Use `INSTRUCTIONS.md` to keep experiments small and measurable.
6. Before every submission, run the checks in `SUBMISSION.md`.

## Root Files

- `POLICY.md` - final-round constraints and allowed tools.
- `INSTRUCTIONS.md` - working rules for local brainstorming agents.
- `TASKS.md` - task router and baseline recipes.
- `LLM_MODELS.md` - local model usage and llama.cpp notes.
- `workspaces/` - per-task memory and handoff notes.

## Minimal Local Setup

- LM Studio with a local model for brainstorming.
- A local code/chat tool if needed, configured to use `http://localhost:1234/v1`.
- No internet, no online LLMs, no external APIs during the final.

