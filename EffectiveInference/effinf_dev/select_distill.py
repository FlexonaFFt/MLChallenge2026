"""Отбор кандидатов self-distill: эвристики (жёсткий фильтр) + локальный судья.

Вход: distill_cand_*.jsonl от gen_distill.py ({id, query, reference, candidates}).
Выход: RAFT-набор в формате messages (minimal system) для дообучения kaggle_sft.py.

Логика (комбо, как договорились):
  1. ЭВРИСТИКИ — жёсткий фильтр + грубый скор каждого кандидата:
     - отсев пустых, обрезанных «на полуслове», с битым LaTeX (нечётные $),
       с протёкшим <think>, с грубым зацикливанием (повтор строк);
     - длина в разумном коридоре относительно эталона;
     - бонус за «**Ответ:**», markdown-структуру, баланс формул.
  2. СУДЬЯ — среди выживших берём top-K по эвристике и сравниваем pairwise с
     эталоном (Qwen2.5-3B vLLM, как judge_local.py). Цель — выбрать ответ,
     который судья ставит НЕ НИЖЕ эталона.
  3. ЦЕЛЬ ДЛЯ RAFT: если лучший кандидат выигрывает/ничья у эталона → берём
     кандидата (модель учится на «выше эталона»); иначе оставляем эталон
     (без регрессии). Доля замен печатается — это и есть сигнал, что distill
     реально что-то добавил, а не шумит.

    python select_distill.py \
        --candidates data/distill_cand_4b.jsonl \
        --out data/raft_4b.jsonl \
        --judge_model Qwen/Qwen2.5-3B-Instruct --topk 2

--no_judge → только эвристики (best по скору, замена если скор > скора эталона).
"""
import argparse
import json
import random
import re

MINIMAL_SYSTEM = "Отвечай на языке вопроса."

_THINK = re.compile(r"<think>|</think>", re.I)
_SENT_END = tuple(".!?…)»\"'`")  # допустимые «закрывающие» символы (не обрыв)


def is_truncated(text: str) -> bool:
    """Грубая эвристика обрыва: не заканчивается завершающим символом/формулой."""
    t = text.rstrip()
    if not t:
        return True
    if t.endswith("$") or t.endswith("$$"):
        return False
    return not t.endswith(_SENT_END)


def has_repetition(text: str, thresh: int = 4) -> bool:
    """Зацикливание: одна и та же непустая строка повторяется >= thresh раз."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    from collections import Counter
    return max(Counter(lines).values()) >= thresh


def hard_ok(cand: str, ref: str) -> bool:
    if not cand or len(cand) < 3:
        return False
    if _THINK.search(cand):
        return False
    if cand.count("$") % 2 != 0:              # незакрытая формула
        return False
    if is_truncated(cand):
        return False
    if has_repetition(cand):
        return False
    rl = max(len(ref), 1)
    if not (0.35 * rl <= len(cand) <= 3.0 * rl):  # длина в коридоре эталона
        return False
    return True


def heuristic_score(cand: str, ref: str) -> float:
    s = 0.0
    if "**Ответ:**" in cand or re.search(r"\bОтвет\s*:", cand):
        s += 1.0
    if re.search(r"^\s*[-•*]\s", cand, re.M):  # списки → структура
        s += 0.3
    if "$" in cand:
        s += 0.3
    # близость длины к эталону (чем ближе, тем лучше)
    rl, cl = max(len(ref), 1), max(len(cand), 1)
    s += 1.0 - min(abs(cl - rl) / rl, 1.0)
    return s


# --- судья (как judge_local.py, но пакетно для top-K кандидатов на запрос) ---
JUDGE_SYSTEM = "Ты — строгий и беспристрастный эксперт, оценивающий ответы на школьные вопросы."
JUDGE_TEMPLATE = """Вопрос школьника:
{query}

Ответ A:
{a}

Ответ B:
{b}

Сравни ответы по фактической правильности, полноте, ясности и корректности
оформления (формулы, шаги, итоговый ответ). Какой ответ лучше?
Ответь РОВНО одним токеном: A, B или TIE."""


def parse_verdict(text: str, cand_is_a: bool) -> str:
    t = text.strip().upper()
    if "TIE" in t[:5]:
        return "tie"
    first = next((ch for ch in t if ch in "AB"), None)
    if first is None:
        return "tie"
    return "win" if (first == "A") == cand_is_a else "loss"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--judge_model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--topk", type=int, default=2, help="сколько лучших по эвристике судить")
    p.add_argument("--no_judge", action="store_true")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = p.parse_args()

    with open(args.candidates, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    # 1) жёсткий фильтр + эвристический рейтинг
    filtered = []  # (row, [ (cand, score) ... ] отсортировано по score desc)
    for r in rows:
        survivors = [(c, heuristic_score(c, r["reference"]))
                     for c in r["candidates"] if hard_ok(c, r["reference"])]
        survivors.sort(key=lambda x: x[1], reverse=True)
        filtered.append((r, survivors))

    n_total = len(rows)
    n_have_survivor = sum(1 for _, s in filtered if s)
    print(f"запросов: {n_total} | хотя бы 1 кандидат прошёл фильтр: {n_have_survivor} "
          f"({n_have_survivor / max(n_total,1) * 100:.1f}%)")

    targets: dict[int, str] = {}     # id -> выбранный ответ для RAFT
    n_replaced = 0

    if args.no_judge:
        for r, survivors in filtered:
            ref = r["reference"]
            if survivors and survivors[0][1] > heuristic_score(ref, ref):
                targets[r["id"]] = survivors[0][0]
                n_replaced += 1
            else:
                targets[r["id"]] = ref
    else:
        # 2) собираем pairwise-сравнения: top-K выживших vs эталон
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tok = AutoTokenizer.from_pretrained(args.judge_model, use_fast=True)
        prompts, meta = [], []   # meta: (row_id, cand_text, cand_is_a)
        for r, survivors in filtered:
            ref = r["reference"]
            for cand, _ in survivors[: args.topk]:
                rng = random.Random(hash((r["id"], cand[:32])) & 0xFFFFFFFF)
                cand_is_a = rng.random() < 0.5
                a, b = (cand, ref) if cand_is_a else (ref, cand)
                user = JUDGE_TEMPLATE.format(query=r["query"], a=a, b=b)
                prompts.append(tok.apply_chat_template(
                    [{"role": "system", "content": JUDGE_SYSTEM},
                     {"role": "user", "content": user}],
                    tokenize=False, add_generation_prompt=True))
                meta.append((r["id"], cand, cand_is_a))

        print(f"сравнений судье: {len(prompts)}")
        llm = LLM(model=args.judge_model, dtype=args.dtype,
                  max_model_len=args.max_model_len,
                  tensor_parallel_size=args.tensor_parallel_size,
                  gpu_memory_utilization=args.gpu_memory_utilization, seed=0)
        sp = SamplingParams(temperature=0.0, max_tokens=4, top_k=-1)
        outputs = llm.generate(prompts, sampling_params=sp)

        # лучший вердикт на запрос: win > tie > (нет → эталон)
        rank = {"win": 2, "tie": 1, "loss": 0}
        best: dict[int, tuple[int, str]] = {}   # id -> (rank, cand)
        for (rid, cand, cand_is_a), o in zip(meta, outputs):
            v = parse_verdict(o.outputs[0].text, cand_is_a)
            cur = best.get(rid)
            if cur is None or rank[v] > cur[0]:
                best[rid] = (rank[v], cand)

        for r, _ in filtered:
            ref = r["reference"]
            b = best.get(r["id"])
            if b is not None and b[0] >= 1:      # win или tie с эталоном
                targets[r["id"]] = b[1]
                n_replaced += 1
            else:
                targets[r["id"]] = ref

    print(f"заменено эталонов кандидатами: {n_replaced}/{n_total} "
          f"({n_replaced / max(n_total,1) * 100:.1f}%)")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            ans = targets[r["id"]]
            rec = {"messages": [
                {"role": "system", "content": MINIMAL_SYSTEM},
                {"role": "user", "content": r["query"]},
                {"role": "assistant", "content": ans},
            ]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"RAFT-набор → {args.out}")


if __name__ == "__main__":
    main()
