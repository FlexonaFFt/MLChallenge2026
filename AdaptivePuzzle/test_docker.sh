#!/bin/bash
# Local CI replica: docker build → train → solve → check
# Reproduces the real checking system as closely as possible.
#
# Usage: ./test_docker.sh <task> [num_instances]
#   task          — ENV_ID value: game_15_2d | toggle_lights | cylinder_game
#   num_instances — puzzles to generate (default: 10)
#
# Env vars (все прокидываются в Docker как есть):
#   TRAIN_TIME_LIMIT  секунды (не задан по умолчанию → train.py использует свой дефолт 600)
#   SOLVE_TIME_LIMIT  секунды (не задан по умолчанию → solve.py использует свой дефолт 300)
#   TRAIN_CPUS        (default: 4)    CI uses 8
#   TRAIN_MEM         (default: 8g)   CI uses 32g
#   SOLVE_CPUS        (default: 4)    CI uses 8
#   SOLVE_MEM         (default: 8g)   CI uses 32g
#   PIDS_LIMIT        (default: 4096) CI uses 8192

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TASK="${1:-game_15_2d}"
NUM_INSTANCES="${2:-10}"
export ENV_ID="$TASK"

IMAGE_NAME="rl-participant-local:test"
WORK_DIR="${SCRIPT_DIR}/work_test_ci"

TRAIN_CPUS="${TRAIN_CPUS:-4}"
TRAIN_MEM="${TRAIN_MEM:-8g}"
SOLVE_CPUS="${SOLVE_CPUS:-4}"
SOLVE_MEM="${SOLVE_MEM:-8g}"
PIDS_LIMIT="${PIDS_LIMIT:-4096}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

fix_ownership() {
    # Same alpine trick as fix_workspace_ownership.sh in CI:
    # Docker runs as root inside the container and leaves root-owned files;
    # we chown them back to the current user.
    docker run --rm \
        -v "${SCRIPT_DIR}:/repo" \
        alpine:3.19 \
        chown -R "$(id -u):$(id -g)" "/repo/work_test_ci" \
        2>/dev/null \
        || sudo chown -R "$(id -u):$(id -g)" "$WORK_DIR" \
        || true
}

# ---------------------------------------------------------------------------
# Step 1: Clean up previous run (root-owned files need alpine to remove)
# ---------------------------------------------------------------------------
echo "=== Cleaning up previous test run ==="
if [ -d "$WORK_DIR" ]; then
    docker run --rm \
        -v "${SCRIPT_DIR}:/repo" \
        alpine:3.19 \
        rm -rf /repo/work_test_ci \
        2>/dev/null \
        || sudo rm -rf "$WORK_DIR" \
        || true
fi
rm -rf "$WORK_DIR"

# ---------------------------------------------------------------------------
# Step 2: Build participant Docker image
# ---------------------------------------------------------------------------
echo ""
echo "=== Building participant Docker image ==="
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Step 3: Prepare workspace (mirrors CI before_script)
# ---------------------------------------------------------------------------
echo ""
echo "=== Preparing workspace (task=${TASK}) ==="
mkdir -p "${WORK_DIR}/weights"

# gym.py is placed in /workspace so the participant container picks it up
# via PYTHONPATH=/workspace:/opt/participant (set in the Dockerfile).
cp "${SCRIPT_DIR}/gym.py" "${WORK_DIR}/gym.py"

# CI does chmod -R a+rwx on the task work dir so Docker (running as root
# inside the container) can write files back to the mounted volume.
chmod -R a+rwx "${WORK_DIR}"

# ---------------------------------------------------------------------------
# Step 4: Generate test instances
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

chmod -R a+rwx "${WORK_DIR}"

# ---------------------------------------------------------------------------
# Step 5: TRAIN
# ---------------------------------------------------------------------------
echo ""
echo "=== TRAIN PHASE ==="
docker run --rm \
    --network none \
    --cap-drop=ALL \
    --security-opt no-new-privileges \
    --env-file <(env) \
    -e PYTHONDONTWRITEBYTECODE=1 \
    --cpus "${TRAIN_CPUS}" \
    --memory "${TRAIN_MEM}" \
    --memory-swap "${TRAIN_MEM}" \
    --pids-limit "${PIDS_LIMIT}" \
    -v "${WORK_DIR}:/workspace:rw" \
    -w /workspace \
    "$IMAGE_NAME" \
    python3 /opt/participant/train.py

echo "=== Fixing workspace ownership after train ==="
fix_ownership

# ---------------------------------------------------------------------------
# Step 6: SOLVE
# ---------------------------------------------------------------------------
echo ""
echo "=== SOLVE PHASE ==="
# CI re-applies chmod before solve too
chmod -R a+rwx "${WORK_DIR}"

docker run --rm \
    --network none \
    --cap-drop=ALL \
    --security-opt no-new-privileges \
    --env-file <(env) \
    -e PYTHONDONTWRITEBYTECODE=1 \
    --cpus "${SOLVE_CPUS}" \
    --memory "${SOLVE_MEM}" \
    --memory-swap "${SOLVE_MEM}" \
    --pids-limit "${PIDS_LIMIT}" \
    -v "${WORK_DIR}:/workspace:rw" \
    -w /workspace \
    "$IMAGE_NAME" \
    python3 /opt/participant/solve.py

echo "=== Fixing workspace ownership after solve ==="
fix_ownership

if [ ! -f "${WORK_DIR}/output_actions.csv" ]; then
    echo "ERROR: solve.py did not produce output_actions.csv"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 7: CHECK (mirrors checker container — gym.py + input + CSV in one dir)
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
# Step 8: Results
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
