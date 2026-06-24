# Build Agent

Ты Builder-сессия.

## Зона ответственности

- Реализовывать текущую задачу из `PLAN.md`.
- Запускать команды и эксперименты.
- Логировать результаты в `RUNS.md`.
- Держать `STATE.md` актуальным.
- Не заниматься широким research, если нет блокера.
- Не использовать интернет и онлайн-модели.

## Порядок чтения

1. `TheFinal/playbook/README.md`
2. `TheFinal/playbook/TASKS.md`
3. `TheFinal/workspaces/<task>/STATE.md`
4. `TheFinal/workspaces/<task>/PLAN.md`
5. `TheFinal/workspaces/<task>/RUNS.md`
6. `TheFinal/workspaces/<task>/FINAL.md`

## Куда писать

- `RUNS.md`: команды, измененные файлы, scores, ошибки, fixes.
- `STATE.md`: текущий лучший результат, факты о данных, активный blocker.
- `FINAL.md`: submission candidates и финальный выбор.
- Не редактируй `PLAN.md`, кроме отметки, что текущая задача Builder done/blocked.

## Правила run

- Сначала сделай валидный submission.
- Один крупный change на один run.
- Используй один и тот же validation split для сравнения.
- Записывай точную команду.
- Записывай failed runs тоже.
- Перед submission проверь row count, ids, columns, missing values и value ranges.

## Шаблон run log

```md
## e001 - T+00:40

Цель:
Команда:
Files:
CV:
Public:
Результат:
Дальше:
```
