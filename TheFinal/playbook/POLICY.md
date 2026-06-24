# Policy

## Hard Rules

- Internet is forbidden during the final.
- Online LLMs and external APIs are forbidden.
- Do not use OpenAI, Anthropic, Gemini, OpenRouter, Hugging Face Inference API, web search, or remote notebooks.
- Use only local files, local docs, local models, competition VM tools, and allowed offline materials.
- If unsure whether a tool contacts the internet, do not use it until it is tested offline.

## Allowed Tools

- JupyterLab on the competition VM.
- `src code` on the VM.
- Local LLMs on the VM or laptop.
- Local documentation, templates, snippets, and previous offline notes.
- Local model weights provided by organizers or prepared before the final.
- Docker image/libraries provided by organizers.

## VM Workflow

- Open JupyterLab.
- Open a terminal tab.
- Run:

```bash
src code
```

- Use the VM for implementation, training, inference, and submission preparation.

## Laptop Workflow

- Use LM Studio or another local-only tool for brainstorming.
- Keep notes in `workspaces/<task_name>/`.
- Transfer only useful plans, prompts, and handoff notes to the VM.

## Safety Checks

- Test local tools with Wi-Fi disabled before the final.
- Prefer `localhost` endpoints only.
- Do not install packages during the final unless explicitly provided offline.
- Do not let agents download models, docs, packages, examples, or benchmarks.

