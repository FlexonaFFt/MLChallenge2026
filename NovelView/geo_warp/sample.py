"""Load a sample and build FLOW-based inputs for the refiner.

Pivot rationale: with only ~0.5 s between t0/t1 the scene moves little, and the
aggregated lidar is contaminated by moving objects + holes, so geometric warping
hurt (12.8 dB < mean 20.0 dB). DIS optical-flow interpolation beats the mean
(~20.8 dB), so we use it as the refiner's base and let the net sharpen it.

Channel layout (see model.UNetRefiner._base_blend):
  raw_t0 0:3 | raw_t1 3:6 | warp_t0 6:9 | warp_t1 9:12 | flow_interp 12:15 | flowmag 15
Base = flow_interp (channels 12:15).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CAMERAS = ["front", "left_fwd", "left_bwd", "right_fwd", "right_bwd", "rear"]


def load_meta(sample_dir: Path) -> dict:
    return json.loads((sample_dir / "meta.json").read_text())


def _read_img(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def get_alpha(meta: dict) -> float:
    """Relative position of the target time in [t0, t1]."""
    ts = meta.get("timestamps_ns")
    if ts and all(k in ts for k in ("t0", "t1", "target")):
        return float((ts["target"] - ts["t0"]) / (ts["t1"] - ts["t0"]))
    return 0.5


def _warp(img: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(
        img, (xs + dx).astype(np.float32), (ys + dy).astype(np.float32),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


_DIS = None


def _dis_flow(g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
    global _DIS
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _DIS.calc(g0, g1, None)


def build_geo_inputs(sample_dir: Path) -> dict:
    meta = load_meta(sample_dir)
    cam = meta["target_camera"]
    a = get_alpha(meta)

    img0 = _read_img(sample_dir / "input" / "t0" / f"{cam}.jpg")
    img1 = _read_img(sample_dir / "input" / "t1" / f"{cam}.jpg")
    H, W = img0.shape[:2]

    g0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    flow = _dis_flow(g0, g1)  # t0 -> t1, (H,W,2)
    dx, dy = flow[..., 0], flow[..., 1]

    f0 = img0.astype(np.float32) / 255.0
    f1 = img1.astype(np.float32) / 255.0
    warp0 = _warp(f0, -a * dx, -a * dy)            # t0 -> mid
    warp1 = _warp(f1, (1 - a) * dx, (1 - a) * dy)  # t1 -> mid
    flow_interp = (1 - a) * warp0 + a * warp1

    out = {
        "raw_t0": f0,
        "raw_t1": f1,
        "warp_t0": warp0,
        "warp_t1": warp1,
        "flow_interp": flow_interp,
        "flow": flow.astype(np.float32),
        "alpha": a,
        "camera": cam,
        "size": (H, W),
    }
    gt_path = sample_dir / "target" / f"{cam}.jpg"
    out["target"] = _read_img(gt_path).astype(np.float32) / 255.0 if gt_path.exists() else None
    return out


def stack_channels(geo: dict) -> np.ndarray:
    """Pack geo dict into a (16,H,W) float32 tensor for the network."""
    flow = geo["flow"]
    mag = np.sqrt((flow ** 2).sum(-1))
    mag = (mag / (mag.max() + 1e-6)).astype(np.float32)  # normalized flow magnitude
    chans = [
        geo["raw_t0"].transpose(2, 0, 1),        # 3
        geo["raw_t1"].transpose(2, 0, 1),        # 3
        geo["warp_t0"].transpose(2, 0, 1),       # 3
        geo["warp_t1"].transpose(2, 0, 1),       # 3
        geo["flow_interp"].transpose(2, 0, 1),   # 3
        mag[None],                               # 1
    ]
    return np.concatenate(chans, axis=0)  # (16, H, W)


IN_CHANNELS = 16
