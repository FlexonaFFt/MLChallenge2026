#!/usr/bin/env bash
# Пайплайн дообучения студента: stage 0 (данные) → 1 (SFT) → 2 (merge + AWQ).
# Параметризован по BASE_MODEL, чтобы прогнать ОБА трека последовательно тем же
# кодом (сначала 4B, потом 1.7B):
#
#   BASE_MODEL=Qwen/Qwen3-4B   TAG=qwen3_4b   bash run_pipeline.sh
#   BASE_MODEL=Qwen/Qwen3-1.7B TAG=qwen3_1p7b bash run_pipeline.sh
#
# Запускается внутри training-образа (см. Dockerfile.train). Чекпойнты и
# готовые AWQ-веса складываются в $WORK (persistent volume).
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"
VARIANT="${VARIANT:-minimal}"        # minimal | rich (см. prepare_data.py)
TAG="${TAG:-qwen3_4b}"
WORK="${WORK:-/work}"
DATA="${DATA:-data}"
EPOCHS="${EPOCHS:-2}"
MAX_LEN="${MAX_LEN:-2048}"           # НЕ занижать: эталоны бывают >1700 токенов
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"   # напр. "--load_4bit" для 7-8B-учителя

mkdir -p "$WORK"
echo "=== BASE_MODEL=$BASE_MODEL  TAG=$TAG  VARIANT=$VARIANT  EPOCHS=$EPOCHS ==="

# ---- stage 0: данные (CPU, один раз) ----
if [ ! -f "$DATA/train_${VARIANT}.jsonl" ]; then
  echo "[stage 0] prepare_data"
  python prepare_data.py
else
  echo "[stage 0] данные уже готовы — пропуск"
fi

# ---- stage 1: SFT (QLoRA/LoRA, masked-loss по ответу) ----
echo "[stage 1] SFT → $WORK/lora_${TAG}"
python train_lora.py \
  --variant "$VARIANT" \
  --base_model "$BASE_MODEL" \
  --train_jsonl "$DATA/train_${VARIANT}.jsonl" \
  --eval_jsonl "$DATA/eval.jsonl" \
  --output_dir "$WORK/lora_${TAG}" \
  --epochs "$EPOCHS" \
  --max_len "$MAX_LEN" \
  $EXTRA_TRAIN_ARGS

# ---- stage 2: merge LoRA → fp16, затем AWQ-квант для L4 ----
echo "[stage 2a] merge → $WORK/merged_${TAG}"
python merge_lora.py \
  --base_model "$BASE_MODEL" \
  --adapter "$WORK/lora_${TAG}" \
  --out "$WORK/merged_${TAG}"

echo "[stage 2b] AWQ-quant → $WORK/awq_${TAG}"
python quantize_awq.py \
  --model "$WORK/merged_${TAG}" \
  --out "$WORK/awq_${TAG}" \
  --calib_jsonl "$DATA/train_${VARIANT}.jsonl"

echo "=== ГОТОВО. Студент: $WORK/awq_${TAG} ==="
echo "Дальше: gen_candidates.py (студент) → judge_local.py (Qwen-судья) для оценки на held-out,"
echo "и только при выигрыше vs текущий конфиг — копировать в weights/ посылки."
