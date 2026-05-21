#!/bin/bash
# Local run without Docker: generate → train → solve → check
# Useful for quick iteration; no resource limits, no network isolation.
#
# Usage: ./test_local.sh <task> [num_instances]
#   task          — ENV_ID value: game_15_2d | toggle_lights | cylinder_game
#   num_instances — puzzles to generate (default: 5)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TASK="${1:-game_15_2d}"
NUM_INSTANCES="${2:-5}"
export ENV_ID="$TASK"

WORK_DIR="${SCRIPT_DIR}/work_test_local"

# ---------------------------------------------------------------------------
# Step 1: Prepare workspace
# ---------------------------------------------------------------------------
echo "=== Preparing workspace (task=${TASK}) ==="
rm -rf "$WORK_DIR"
mkdir -p "${WORK_DIR}/weights"

cp "${SCRIPT_DIR}/gym.py" "${WORK_DIR}/gym.py"

# ---------------------------------------------------------------------------
# Step 2: Generate test instances
# ---------------------------------------------------------------------------
echo ""
echo "=== Generating test instances ==="

cd "$SCRIPT_DIR"
PYTHONPATH="${WORK_DIR}:${SCRIPT_DIR}" \
    python3 "${SCRIPT_DIR}/generate_states.py" \
        --num_instances "$NUM_INSTANCES" \
        --public_output "${WORK_DIR}/input_states.jsonl"

if [ ! -f "${WORK_DIR}/input_states.jsonl" ]; then
    echo "ERROR: generate_states.py did not produce input_states.jsonl"
    exit 1
fi
echo "Generated $(wc -l < "${WORK_DIR}/input_states.jsonl") test instances"

# ---------------------------------------------------------------------------
# Step 3: TRAIN
# ---------------------------------------------------------------------------
echo ""
echo "=== TRAIN PHASE ==="
cd "$WORK_DIR"
PYTHONPATH="${WORK_DIR}:${SCRIPT_DIR}" \
    python3 "${SCRIPT_DIR}/train.py"

# ---------------------------------------------------------------------------
# Step 4: SOLVE
# ---------------------------------------------------------------------------
echo ""
echo "=== SOLVE PHASE ==="
cd "$WORK_DIR"
PYTHONPATH="${WORK_DIR}:${SCRIPT_DIR}" \
    python3 "${SCRIPT_DIR}/solve.py"

if [ ! -f "${WORK_DIR}/output_actions.csv" ]; then
    echo "ERROR: solve.py did not produce output_actions.csv"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 5: CHECK
# ---------------------------------------------------------------------------
echo ""
echo "=== CHECK PHASE ==="
CHECK_DIR="${WORK_DIR}/check"
mkdir -p "$CHECK_DIR"

cp "${SCRIPT_DIR}/gym.py"               "${CHECK_DIR}/gym.py"
cp "${WORK_DIR}/input_states.jsonl"     "${CHECK_DIR}/input_states.jsonl"
cp "${WORK_DIR}/output_actions.csv"     "${CHECK_DIR}/output_actions.csv"

cd "$CHECK_DIR"
PYTHONPATH="${CHECK_DIR}" \
    python3 "${SCRIPT_DIR}/check.py" \
        --input input_states.jsonl \
        --submission output_actions.csv

# ---------------------------------------------------------------------------
# Step 6: Results
# ---------------------------------------------------------------------------
echo ""
echo "=== RESULTS ==="
echo -n "Verdict: "
cat "${CHECK_DIR}/verdict.txt" 2>/dev/null || echo "(missing)"
echo ""
if [ -f "${CHECK_DIR}/score.json" ]; then
    cat "${CHECK_DIR}/score.json"
else
    echo "(no score.json)"
fi
echo ""
