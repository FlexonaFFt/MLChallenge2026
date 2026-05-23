"""Узкий sanity-check точности по \\boxed{...} (без модели, бесплатно).

    python auto_check.py --candidates data/cand_minimal.jsonl

ВНИМАНИЕ: покрывает только ~10% вопросов (где у эталона есть \\boxed{}),
и сравнение строкой — приблизительное. Это вспомогательный объективный
сигнал способности на математике, НЕ основная метрика. Основной офлайн-сигнал
качества — judge_local.py (работает на прозе, которой тут большинство).
"""
import argparse
import json
import re


def extract_boxed(text: str) -> str | None:
    """Содержимое последнего \\boxed{...} с учётом вложенных скобок."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out) if depth == 0 else None


def final_answer(text: str) -> str | None:
    """Только \\boxed{} — это единственный чистый источник значения."""
    return extract_boxed(text)


_FRAC = re.compile(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}")
_TEXT = re.compile(r"\\text\{[^{}]*\}")


def normalize(ans: str) -> str:
    s = ans.strip()
    s = _TEXT.sub("", s)                      # убрать единицы \text{кг} и т.п.
    s = _FRAC.sub(r"\1/\2", s)                # \frac{a}{b} -> a/b
    for tok in ["$", "**", "\\(", "\\)", "\\[", "\\]", "\\,", "\\!",
                "\\ ", "\\left", "\\right", " ", "{", "}"]:
        s = s.replace(tok, "")
    s = s.rstrip(".")
    s = s.replace(",", ".")                    # десятичная запятая
    return s.lower()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--eval_jsonl", default="data/eval.jsonl")
    p.add_argument("--dump_misses", default="", help="опц. jsonl с расхождениями")
    args = p.parse_args()

    refs = {}
    with open(args.eval_jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            refs[r["rid"]] = r["reference"]

    with open(args.candidates, encoding="utf-8") as f:
        cands = [json.loads(line) for line in f]

    checked = correct = 0
    misses = []
    for c in cands:
        ref_ans = final_answer(refs[c["rid"]])
        if ref_ans is None:
            continue  # у эталона нет однозначного ответа — пропускаем
        checked += 1
        cand_ans = final_answer(c["answer"])
        ok = cand_ans is not None and normalize(cand_ans) == normalize(ref_ans)
        if ok:
            correct += 1
        elif args.dump_misses:
            misses.append({"rid": c["rid"], "ref": ref_ans, "got": cand_ans})

    cov = checked / len(cands) if cands else 0
    acc = correct / checked if checked else 0
    print(f"\n=== {args.candidates} ===")
    print(f"проверяемых (есть ответ у эталона): {checked}/{len(cands)} ({cov*100:.0f}%)")
    print(f"точность на них: {correct}/{checked} = {acc*100:.1f}%")

    if args.dump_misses and misses:
        with open(args.dump_misses, "w", encoding="utf-8") as f:
            for m in misses:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"расхождения → {args.dump_misses}")


if __name__ == "__main__":
    main()
