# Бейзлайн

Учим маленькую `V(s)` оценивать, сколько шагов осталось до решённого,
и используем её эвристикой в A* (`f = g + V`).

Сеть про конкретную игру ничего не знает: ходит через `env.encode_state`,
`env.valid_actions`, `env.step`, поэтому одна и та же модель работает
на всех играх из `gym.py`.


## Файлы

Файлы задачи:
- `gym.py` — три головоломки, выбираются через `ENV_ID`:
  - `game_15_2d` — пятнашки;
  - `toggle_lights` — погасить все лампочки, нажатие переключает строку и столбец;
  - `cylinder_game` — варикон / советский цилиндр-светофор.
- `generate_states.py` — генерирует `input_states.jsonl` так же, как организаторы.
- `check.py` — считает скор по `output_actions.csv`, пишет `verdict.txt` и `score.json`.

Файлы решения:
- `common.py` — токенизация состояния + случайные блуждания от решённого
  для сбора данных.
- `model.py` — `ValueNet`: эмбеддинги значений, MLP по токенам, mean+max
  pool, маленькая голова, `softplus` на выходе.
- `search.py` — A* с батчевой оценкой детей через `V`.
- `train.py` — собирает датасет, обучает `V`, сохраняет `model.pt`.
- `solve.py` — грузит `model.pt`, запускает A* на каждом инстансе,
  пишет `output_actions.csv`.
- `Dockerfile` — `python:3.11-slim` + `numpy` + `torch`.


# Быстрый запуск (без Docker)

```bash
./test_local.sh <головоломка> <количество задач>
```
Например,
```bash
./test_local.sh toggle_lights 10
```


Для быстрого теста:
```bash
TRAIN_TIME_LIMIT=60 SOLVE_TIME_LIMIT=30 ./test_local.sh toggle_lights 2
```

# Запуск с docker, как в тестирующей системе

```bash
./test_docker.sh <головоломка> <количество задач>
```

## API среды

Все игры реализуют `BasePuzzleEnv` из `gym.py`:

- `reset(seed=None)` — вернуть среду в решённое состояние.
- `get_state()` / `set_state(state)` — снимок состояния, JSON-сериализуемый через `gym.to_jsonable`.
- `solved_state()`, `is_solved()` — цель и проверка.
- `valid_actions()` — список действий, допустимых сейчас (`"X+"`, `"3_4"`, `"L_2"`, …).
- `step(action)` → `(state, reward, done, info)`.
- `inverse_action(action)`, `scramble(length, seed, no_backtrack=True)`.
- `encode_state(state=None)` — dict с `positions` (Nx3), `content_types`/`content_values`, `target_types`/`target_values`.
- `encode_actions(actions=None, state=None)` — структурное описание действий

## Как работает

**Данные.** `backward_walks` стартует из решённого состояния и делает
случайные шаги до глубины `max_walk`. Каждый шаг даёт пару (состояние,
номер шага) — оценка расстояния до цели.

**Представление.** Каждая ячейка превращается в вектор из 15 чисел:
позиция в 3D, тип и индекс содержимого, тип и индекс цели, флаги
совпадения. Размер словаря значений зафиксирован — формат один и тот
же для всех игр, сетка не зависит от количества клеток.

**Сеть.** Per-token MLP → mean+max pool → MLP-голова → `softplus`.

**A\*.** Классический алгоритм с использованием эвристики V(s) от модели

## Запуск

```bash
# выбрать игру в gym.py: ENV_ID = "game_15_2d" / "toggle_lights" / "cylinder_game"
python generate_states.py
python train.py
python solve.py
python check.py
cat verdict.txt
cat score.json
```

Для быстрой проверки локально все скрипты принимают `--time_limit`
в секундах:

```bash
python train.py --time_limit 60
python solve.py --time_limit 60
```
