import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from source.config import InferenceConfig

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


class SchoolQAEngine:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_dir, use_fast=True)
        self.retriever = None
        if config.dynamic_few_shot:
            try:
                from source.retriever import FewShotRetriever, default_pool_path
                self.retriever = FewShotRetriever(default_pool_path())
            except Exception as e:  # пул/sklearn недоступны → откатываемся на статику
                print(f"[retriever disabled: {e}] fallback to static few-shot")
        self.llm = LLM(
            model=config.model_dir,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            tokenizer_mode="auto",
            enable_prefix_caching=config.enable_prefix_caching,
            max_num_seqs=config.max_num_seqs,
            seed=0,
        )
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            top_k=-1,
        )

    def _few_shot_for(self, question: str) -> list[tuple[str, str]]:
        """Динамический few-shot (похожие из пула) или статический фолбэк."""
        if self.retriever is not None:
            hits = self.retriever.topk(
                question,
                k=self.config.few_shot_k,
                max_answer_chars=self.config.few_shot_max_answer_chars,
            )
            if hits:
                return hits
        return self.config.few_shot

    def _build_prompts(self, questions: list[str]) -> list[str]:
        prompts = []
        for question in questions:
            few_shot_msgs = []
            for q, a in self._few_shot_for(question):
                few_shot_msgs.append({"role": "user", "content": q})
                few_shot_msgs.append({"role": "assistant", "content": a})
            prompts.append(
                self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": self.config.system_prompt}]
                    + few_shot_msgs
                    + [{"role": "user", "content": question}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        return prompts

    def generate(self, rows: list[dict]) -> list[dict]:
        prompts = self._build_prompts([row["question"] for row in rows])
        outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)
        return [
            {"rid": row["rid"], "answer": out.outputs[0].text.strip()}
            for row, out in zip(rows, outputs)
        ]
