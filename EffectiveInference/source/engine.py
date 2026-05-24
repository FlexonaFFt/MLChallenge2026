import os
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from source.config import FEW_SHOT_BY_CATEGORY, InferenceConfig

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Классификатор категории по ключевым словам. Порядок = приоритет (специфичные выше).
_CATEGORY_RULES: list[tuple[str, re.Pattern]] = [
    ("morphology", re.compile(r"по составу|корень|суффикс|приставк|окончани|морфемн|фонетическ|звуко-?букв", re.I)),
    ("chemistry", re.compile(r"реакц|оксид|кислот|гидроксид|\bсол[ьи]\b|валентност|H2O|CO2|NaOH|HCl|H2SO4|химическ", re.I)),
    ("english", re.compile(r"present|past|perfect|continuous|английск|translate|complete the|in brackets|[A-Za-z]{4,}\s+[A-Za-z]{4,}", re.I)),
    ("essay", re.compile(r"сочинени|эссе|напиши текст|напиши рассказ|рассужден", re.I)),
    ("grammar", re.compile(r"пропущенн\w* букв|вставь|спишите|раскрой скобк|орфограм|запят|ударени|мягкий знак|род существ", re.I)),
    ("physics", re.compile(r"скорост|ускорен|сопротивлен|сила трения|мощност|\bток\b|напряжен|\bволн|кинетическ|потенциальн|\b(Дж|Вт|Ом|Н)\b", re.I)),
    ("math", re.compile(r"уравнени|вычисли|реши|упрост|разложи|найдите значени|производн|интеграл|треугольник|\bдроб|\d\s*[+\-*/=^]\s*\d|x\^?2|%", re.I)),
]


class SchoolQAEngine:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_dir, use_fast=True)

        self.matcher = None
        if config.enable_shortcut:
            try:
                from source.retriever import ExactMatcher, default_pool_path
                self.matcher = ExactMatcher(default_pool_path())
            except Exception as e:  # пул недоступен → шорткат просто выключается
                print(f"[shortcut disabled: {e}]")

        llm_kwargs: dict = dict(
            model=config.model_dir,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            tokenizer_mode="auto",
            enable_prefix_caching=config.enable_prefix_caching,
            max_num_seqs=config.max_num_seqs,
            seed=0,
        )
        if config.spec_draft_model:
            llm_kwargs["speculative_config"] = {
                "method": "draft_model",
                "model": config.spec_draft_model,
                "num_speculative_tokens": config.num_speculative_tokens,
            }
        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            top_k=-1,
        )

    def _classify(self, question: str) -> str:
        for cat, rx in _CATEGORY_RULES:
            if rx.search(question):
                return cat
        return "default"

    def _few_shot_for(self, question: str) -> list[tuple[str, str]]:
        if self.config.enable_category_fewshot:
            return FEW_SHOT_BY_CATEGORY.get(self._classify(question), FEW_SHOT_BY_CATEGORY["default"])
        return self.config.few_shot

    def _apply_template(self, messages: list[dict]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    def _build_prompt(self, question: str) -> str:
        msgs = [{"role": "system", "content": self.config.system_prompt}]
        for q, a in self._few_shot_for(question):
            msgs.append({"role": "user", "content": q})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": question})
        return self._apply_template(msgs)

    _THINK = re.compile(r"<think>.*?</think>", re.S)
    _PREAMBLE = re.compile(
        r"^\s*(конечно[!,. ]*|хорошо[!,. ]*|вот( и тебе| тебе| вам)?\s*"
        r"(решени\w*|ответ\w*|разбор\w*)?[:!,. ]*|давай(те)?\s+разбер\w*[:!,. ]*)",
        re.I,
    )

    def _postprocess(self, text: str) -> str:
        if not self.config.enable_postprocess:
            return text.strip()
        text = self._THINK.sub("", text)
        text = text.strip()
        # срезать вводную фразу-филлер в начале (не более одного раза)
        m = self._PREAMBLE.match(text)
        if m and m.end() < len(text):
            text = text[m.end():].lstrip()
        return text.strip()

    def generate(self, rows: list[dict]) -> list[dict]:
        answers: dict = {}
        model_rows = []
        for row in rows:
            q = row["question"]
            hit = self.matcher.lookup(q) if self.matcher is not None else None
            if hit is not None:
                answers[row["rid"]] = hit          # точный дубликат → эталон без модели
            else:
                model_rows.append(row)

        if model_rows:
            prompts = [self._build_prompt(r["question"]) for r in model_rows]
            outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)
            for r, out in zip(model_rows, outputs):
                answers[r["rid"]] = self._postprocess(out.outputs[0].text)

        return [{"rid": row["rid"], "answer": answers[row["rid"]]} for row in rows]
