# Task B — Geometric refiner: training on the server

Pipeline: **lidar + poses → backward-warp t0/t1 into the target view → U-Net refiner**.
You submit images (`submission/<id>/pred.jpg`), so there is no inference time limit.

## 0. One-time sanity check (do this first, ~1 min, no GPU)

Confirms the format-sensitive bits (timestamps, distortion, axes) are correct on a
real sample. Look at `preview/*.png` and the printed **GEOMETRIC BASE PSNR** — that's
the no-training floor.

```bash
pip install numpy scipy opencv-python-headless pillow
python preview_geo.py --sample-dir /data/train/<sample_id> --out preview
```

If warps look misaligned or coverage is ~0%, the distortion/axis assumptions in
`geo_warp/geometry.py` (search `# VERIFY on real sample`) need adjusting before training.

## 1. Train (Docker, A5000)

```bash
# DATA must contain  train/  test/
DATA=/data CACHE=/cache EPOCHS=50 BATCH=4 CROP=512 ./run_train.sh
```

This builds `nvs:latest`, launches a **detached** container `nvs-train`, and starts
TensorBoard inside it.

### Watching progress (three ways)

| What | How |
|---|---|
| Live console log | `docker logs -f nvs-train` |
| Persistent logfile | `runs/exp1/train.log` (tail it: `tail -f runs/exp1/train.log`) |
| Curves + image grids | TensorBoard at `http://<server-ip>:6006` (loss, PSNR, lr, base/pred/GT) |

Each step logs: `loss psnr l1 ssim | img/s | ETA`. Each epoch logs train means and a
**val PSNR**. Best-by-val checkpoint is saved to `runs/exp1/best.pt`.

## 2. Generate submission

```bash
DATA=/data CKPT=/workspace/runs/exp1/best.pt ./run_infer.sh
docker logs -f nvs-infer
```

Output: `submission/<sample_id>/pred.jpg`.

## 3. What to keep before stopping the server

1. `runs/exp1/best.pt` (and `last.pt`) — the only learned artifact.
2. `geo_warp/`, `refiner/`, `docker/`, `run_*.sh` — code that produced it.
3. `submission/` — zip and upload this.

Everything else (`/cache`, the dataset, the Docker image) is regenerable.

## Knobs

- `BATCH` / `CROP`: lower if you hit OOM on 24 GB (e.g. `BATCH=2 CROP=384`).
- `--perceptual`: adds a light VGG perceptual term (downloads weights once; needs internet).
- `--base`: U-Net width (default 48). Bigger = stronger but slower / more VRAM.
- Big dataset won't fit 256 GB at once → train on a subset or stream shards into `/data`.
