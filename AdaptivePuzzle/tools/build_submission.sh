#!/bin/bash
# Build the submission zip from ../src.
# The zip contains exactly the files Docker COPYs (flat, no nested dirs),
# which is what the grading system expects as the participant folder.
#
# Usage: ./build_submission.sh [output.zip]   (default: ../submission.zip)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../src" && pwd)"
OUT="${1:-${SCRIPT_DIR}/../submission.zip}"

rm -f "$OUT"
( cd "$SRC_DIR" && zip -j "$OUT" Dockerfile train.py solve.py common.py model.py search.py optimize.py )

echo "=== submission contents ==="
unzip -l "$OUT"
echo "built: $OUT"
