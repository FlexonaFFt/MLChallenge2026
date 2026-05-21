import json
import os
import time
from typing import List, Tuple

import torch

from models.value_net import ValueNet
from core.encoder import StateEncoder
from core.collector import DataCollector
from core.trainer import Trainer
from core.searcher import BidirectionalSearcher


MODEL_PATH = "model.pt"
META_PATH = "meta.json"
SAFETY_MARGIN_TRAIN = 20
SAFETY_MARGIN_SOLVE = 10


def _find_pretrained(env_id: str, model_path: str) -> str | None:
    """Return path to the best available pretrained checkpoint."""
    specific = model_path.replace(".pt", f"_{env_id}.pt")
    if os.path.exists(specific):
        return specific
    if os.path.exists(model_path):
        return model_path
    return None


class Agent:
    def __init__(self, env, model_path: str = MODEL_PATH):
        self.env = env
        self.model_path = model_path
        self.encoder = StateEncoder()
        self.collector = DataCollector(env, self.encoder)

    def _load_pretrained(self, env_id: str) -> ValueNet | None:
        path = _find_pretrained(env_id, self.model_path)
        if path is None:
            return None
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = ValueNet()
        model.load_state_dict(ckpt["state_dict"])
        print(f"loaded pretrained model: {path}")
        return model

    def train(
        self,
        time_limit: int,
        seed: int = 239,
        num_pairs: int = 40000,
        max_walk: int = 60,
        batch_size: int = 256,
        lr: float = 1e-3,
    ):
        start = time.time()
        deadline = start + time_limit - SAFETY_MARGIN_TRAIN
        env_id = getattr(self.env, "env_id", "unknown")

        model = self._load_pretrained(env_id) or ValueNet()

        trainer = Trainer(model, self.collector, self.encoder, batch_size=batch_size, lr=lr)
        history = trainer.fit(deadline, num_pairs=num_pairs, max_walk=max_walk, seed=seed)

        torch.save(
            {"state_dict": model.state_dict(), "config": {"env_id": env_id, "history": history}},
            self.model_path,
        )
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump({"env_id": env_id, "wall_time_sec": time.time() - start}, f, indent=2)

        print(f"train done in {time.time() - start:.1f}s")

    def solve_all(
        self, instances: list, time_limit: int
    ) -> List[Tuple[str, List[str]]]:
        start = time.time()
        deadline = start + time_limit - SAFETY_MARGIN_SOLVE
        env_id = getattr(self.env, "env_id", "unknown")

        model = None
        if os.path.exists(self.model_path):
            ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
            model = ValueNet()
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

        searcher = BidirectionalSearcher(self.env, self.encoder, model)
        n = len(instances)
        results = []

        for i, inst in enumerate(instances):
            iid = inst["instance_id"]
            if time.time() >= deadline:
                results.append((iid, []))
                continue

            remaining = n - i
            inst_deadline = time.time() + max(0.5, (deadline - time.time()) / remaining)

            try:
                sol = searcher.solve(inst["state"], inst_deadline)
            except Exception as e:
                print(f"  {iid} failed: {repr(e)}")
                sol = None

            results.append((iid, sol or []))

            if (i + 1) % 25 == 0:
                solved = sum(1 for _, a in results if a)
                print(f"  {i + 1}/{n} solved={solved} elapsed={time.time() - start:.0f}s")

        solved = sum(1 for _, a in results if a)
        print(f"final: solved {solved}/{n}, time {time.time() - start:.1f}s")
        return results
