#!/usr/bin/env bash
# Generate the submission with a trained checkpoint.
# Watch:  docker logs -f nvs-infer
set -euo pipefail

DATA=${DATA:-/data}
CKPT=${CKPT:-/workspace/runs/exp1/best.pt}
OUT=${OUT:-/workspace/submission}
NAME=${NAME:-nvs-infer}
GPU_ARGS=${GPU_ARGS:---gpus all}

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  $GPU_ARGS --ipc=host \
  -v "$DATA":/data:ro \
  -v "$(pwd)/runs":/workspace/runs \
  -v "$(pwd)/submission":/workspace/submission \
  nvs:latest \
  python -u refiner/infer.py --data /data --split test --ckpt "$CKPT" --out "$OUT"

echo ">> inference started. Follow with:  docker logs -f $NAME"
echo ">> result -> ./submission/<sample_id>/pred.jpg   (this is what you upload)"
