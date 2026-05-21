from dataclasses import dataclass, field

SYSTEM_PROMPT = (
    "Ты — опытный школьный репетитор. "
    "Отвечай точно, структурированно и понятно для школьника. "
    "Отвечай на том языке, на котором задан вопрос."
)


@dataclass
class InferenceConfig:
    model_dir: str = "./weights"
    max_new_tokens: int = 384
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    temperature: float = 0.0
    system_prompt: str = field(default=SYSTEM_PROMPT)
    enable_prefix_caching: bool = True
    max_num_seqs: int = 256
