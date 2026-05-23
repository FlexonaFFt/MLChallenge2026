import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from source.config import InferenceConfig

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


class SchoolQAEngine:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_dir, use_fast=True)
        self.llm = LLM(
            model=config.model_dir,
            dtype="bfloat16",
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

    def _build_prompts(self, questions: list[str]) -> list[str]:
        # system + few-shot примеры (как реальные реплики) + сам вопрос.
        # Общий префикс одинаков для всех запросов → кэшируется prefix caching.
        few_shot_msgs = []
        for q, a in self.config.few_shot:
            few_shot_msgs.append({"role": "user", "content": q})
            few_shot_msgs.append({"role": "assistant", "content": a})
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": self.config.system_prompt}]
                + few_shot_msgs
                + [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for question in questions
        ]

    def generate(self, rows: list[dict]) -> list[dict]:
        prompts = self._build_prompts([row["question"] for row in rows])
        outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)
        return [
            {"rid": row["rid"], "answer": out.outputs[0].text.strip()}
            for row, out in zip(rows, outputs)
        ]
