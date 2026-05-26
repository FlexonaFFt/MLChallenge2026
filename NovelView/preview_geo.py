"""Sanity-check the geometry on ONE real sample before any training.

No torch needed (cv2 + numpy + scipy + pillow). Dumps the two warps, the validity
masks, the geometric base blend and — if a GT target exists — prints the base PSNR.
This is the fastest way to confirm the format-sensitive bits (timestamps,
distortion, axes) are correct.

Run:  python preview_geo.py --sample-dir /data/train/<sample_id> --out preview
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from geo_warp import build_geo_inputs


def psnr_np(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-9 else 20 * np.log10(255.0 / np.sqrt(mse))


def save(arr01, path):
    Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8)).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-dir", required=True, type=Path)
    p.add_argument("--out", default="preview", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    geo = build_geo_inputs(args.sample_dir)
    base = geo["flow_interp"]

    save(geo["warp_t0"], args.out / "warp_t0.png")
    save(geo["warp_t1"], args.out / "warp_t1.png")
    save(base, args.out / "flow_interp.png")

    print(f"camera={geo['camera']} size={geo['size']} alpha={geo['alpha']:.3f}")

    if geo["target"] is not None:
        gt = (geo["target"] * 255).astype(np.uint8)
        base_u8 = (np.clip(base, 0, 1) * 255).astype(np.uint8)
        raw_mean = (np.clip(0.5 * (geo["raw_t0"] + geo["raw_t1"]), 0, 1) * 255).astype(np.uint8)
        save(geo["target"], args.out / "target_gt.png")
        save(0.5 * (geo["raw_t0"] + geo["raw_t1"]), args.out / "raw_mean.png")
        print(f"RAW mean(t0,t1) PSNR vs GT = {psnr_np(raw_mean, gt):.2f} dB")
        print(f"FLOW interp base PSNR      = {psnr_np(base_u8, gt):.2f} dB  <- refiner base")
        print("(refiner starts from flow interp and learns a residual toward GT)")
    print(f"images written to {args.out}/")


if __name__ == "__main__":
    main()
