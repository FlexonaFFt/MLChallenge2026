"""Run the trained refiner over the test split and write submission/<id>/pred.jpg.

Full-frame inference with optional tiling for very large images. Logs progress so
you can watch it on the server the same way as training.

Run:  python -u infer.py --data /data --ckpt runs/exp1/best.pt --out submission
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geo_warp import IN_CHANNELS, build_geo_inputs, stack_channels  # noqa: E402
from refiner.model import UNetRefiner  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("infer")


@torch.no_grad()
def infer_one(model, sample_dir, device, pad_to=64):
    geo = build_geo_inputs(sample_dir)
    x = torch.from_numpy(stack_channels(geo)).unsqueeze(0).to(device)
    _, _, H, W = x.shape
    ph = ((H - 1) // pad_to + 1) * pad_to
    pw = ((W - 1) // pad_to + 1) * pad_to
    x = torch.nn.functional.pad(x, [0, pw - W, 0, ph - H], mode="replicate")
    with torch.cuda.amp.autocast(enabled=device == "cuda"):
        pred = model(x)
    pred = pred[:, :, :H, :W].squeeze(0).clamp(0, 1).cpu().numpy()
    return (pred.transpose(1, 2, 0) * 255).round().astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="submission")
    p.add_argument("--base", type=int, default=48)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    base = ck.get("args", {}).get("base", args.base)
    model = UNetRefiner(in_ch=IN_CHANNELS, base=base).to(device).eval()
    model.load_state_dict(ck["model"])
    log.info(f"loaded {args.ckpt} (best val PSNR={ck.get('best', float('nan')):.3f}) base={base}")

    test_root = Path(args.data) / args.split
    dirs = sorted(d for d in test_root.iterdir() if d.is_dir())
    out_root = Path(args.out)
    log.info(f"{len(dirs)} samples -> {out_root}")

    t0 = time.time()
    for i, sd in enumerate(tqdm(dirs, dynamic_ncols=True)):
        img = infer_one(model, sd, device)
        od = out_root / sd.name
        od.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(od / "pred.jpg", quality=95)
        if (i + 1) % 50 == 0:
            log.info(f"[{i+1}/{len(dirs)}] {(i+1)/(time.time()-t0):.2f} samp/s")
    log.info(f"done. submission written to {out_root}/")


if __name__ == "__main__":
    main()
