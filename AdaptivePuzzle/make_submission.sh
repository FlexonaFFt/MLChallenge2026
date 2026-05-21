#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

OUT="submission.zip"
rm -f "$OUT"

zip -r "$OUT" \
    Dockerfile \
    train.py solve.py gym.py \
    model_game_15_2d.pt model_cylinder_game.pt \
    core/ models/ utils/ \
    --exclude "**/__pycache__/*" \
    --exclude "**/*.pyc" \
    --exclude "**/*.pyo"

echo "Created $OUT ($(du -sh "$OUT" | cut -f1))"
