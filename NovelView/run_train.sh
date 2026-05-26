#!/usr/bin/env bash
# Build the image and launch training in a named, detached container.
# Watch logs live:   docker logs -f nvs-train
# TensorBoard:        open http://<server-ip>:6006   (started below)
#
# Layout on the server (adjust DATA to wherever you unpacked the Kaggle data):
#   DATA  -> dataset root containing  train/  test/
#   CACHE -> fast scratch for precomputed geo stacks (regenerable)
#   OUT   -> run dir: checkpoints + logs + tensorboard
set -euo pipefail

DATA=${DATA:-/data}
CACHE=${CACHE:-/cache}
OUT=${OUT:-/workspace/runs/exp1}
NAME=${NAME:-nvs-train}
# GPU flags: default to modern `--gpus all`. If your host errors with
# "failed to discover GPU vendor from CDI", set:
#   GPU_ARGS="--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all"
GPU_ARGS=${GPU_ARGS:---gpus all}
EPOCHS=${EPOCHS:-50}
BATCH=${BATCH:-4}
CROP=${CROP:-512}

mkdir -p "$CACHE" "$(dirname "$OUT")"

echo ">> building image nvs:latest"
# --network=host: use the host's networking/DNS during build (the build sandbox
# otherwise often can't reach pypi even when the host can).
docker build --network=host -t nvs:latest -f docker/Dockerfile .

echo ">> removing old container if present"
docker rm -f "$NAME" 2>/dev/null || true

echo ">> launching training container '$NAME'"
docker run -d --name "$NAME" \
  $GPU_ARGS --ipc=host \
  -v "$DATA":/data:ro \
  -v "$CACHE":/cache \
  -v "$(pwd)/runs":/workspace/runs \
  -p 6006:6006 \
  nvs:latest \
  bash -lc "tensorboard --logdir /workspace/runs --host 0.0.0.0 --port 6006 >/workspace/runs/tb.log 2>&1 & \
            python -u refiner/train.py \
              --data /data --cache-dir /cache --out $OUT \
              --epochs $EPOCHS --batch $BATCH --crop $CROP --workers 8 --amp"

echo
echo ">> training started. Follow logs with:"
echo "     docker logs -f $NAME"
echo ">> TensorBoard at http://<server-ip>:6006"
echo ">> checkpoints will appear in ./runs/$(basename "$OUT")/  (last.pt, best.pt)"
