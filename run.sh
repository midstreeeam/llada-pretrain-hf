#!/usr/bin/env bash

# Prevent sourcing: run as an executable script so shell options don't leak.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "[error] Please run this script instead of sourcing it: bash run.sh" >&2
  return 1
fi

set -euo pipefail

# Determine repository root so the script is location-independent.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate local virtual environment if it exists.
if [[ -d "${REPO_DIR}/.venv" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_DIR}/.venv/bin/activate"
else
  echo "[warn] No virtual environment at ${REPO_DIR}/.venv; falling back to current interpreter."
fi

# ===== CONFIGURATION =====
MODEL_PATH="${MODEL_PATH:-answerdotai/ModernBERT-base}"
DATASET_NAME="${DATASET_NAME:-tinystories}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/model_config/llada_100m.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/output/llada_100m_tinystories}"
MODE="${MODE:-llada}"

# ===== HYPERPARAMETERS =====
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MLM_SCHEDULE_TYPE="${MLM_SCHEDULE_TYPE:-cosine}"
MLM_PROB_START="${MLM_PROB_START:-0.3}"
MLM_PROB_END="${MLM_PROB_END:-0.15}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LOGGING_STEPS="${LOGGING_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-20000}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-1}"
GENERATION_INTERVAL="${GENERATION_INTERVAL:-100}"
GENERATION_MAX_NEW_TOKENS="${GENERATION_MAX_NEW_TOKENS:-64}"
GENERATION_DIFFUSION_STEPS="${GENERATION_DIFFUSION_STEPS:-16}"
GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-1.0}"
GENERATION_BLOCK_SIZE="${GENERATION_BLOCK_SIZE:-8}"
GENERATION_DECODE_TOP_K="${GENERATION_DECODE_TOP_K:-0}"
GENERATION_DO_SAMPLE_FLAG=""
case "${GENERATION_DO_SAMPLE:-}" in
  1|true|TRUE|True|yes|YES|on|ON)
    GENERATION_DO_SAMPLE_FLAG="--generation_do_sample"
    ;;
esac
GENERATION_DEBUG_FLAG=""
case "${GENERATION_DEBUG:-}" in
  1|true|TRUE|True|yes|YES|on|ON)
    GENERATION_DEBUG_FLAG="--generation_debug"
    ;;
esac

# Ensure the repo is importable when running main.py directly.
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_DIR}"

EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM_STEPS))

echo "========================================="
echo "LLaDA 100M Training on TinyStories"
echo "========================================="
echo "Model path: ${MODEL_PATH}"
echo "Model config: ${CONFIG_PATH}"
echo "Dataset: ${DATASET_NAME}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Mode: ${MODE}"
echo "Effective batch size: ${EFFECTIVE_BATCH} per GPU"
echo "========================================="
echo ""

set +e
python3 "${REPO_DIR}/main.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --validation_dataset_name "none" \
  --output_dir "${OUTPUT_DIR}" \
  --config_path "${CONFIG_PATH}" \
  --mode "${MODE}" \
  --num_train_epochs "${EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --max_length "${MAX_LENGTH}" \
  --mlm_start_prob "${MLM_PROB_START}" \
  --mlm_end_prob "${MLM_PROB_END}" \
  --mlm_schedule_type "${MLM_SCHEDULE_TYPE}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --logging_steps "${LOGGING_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --seed "${SEED}" \
  --dataloader_num_workers "${NUM_WORKERS}" \
  --bf16 \
  --generation_interval "${GENERATION_INTERVAL}" \
  --generation_max_new_tokens "${GENERATION_MAX_NEW_TOKENS}" \
  --generation_num_diffusion_steps "${GENERATION_DIFFUSION_STEPS}" \
  --generation_temperature "${GENERATION_TEMPERATURE}" \
  --generation_block_size "${GENERATION_BLOCK_SIZE}" \
  --generation_decode_top_k "${GENERATION_DECODE_TOP_K}" \
  --generation_prompts "Once upon a time, a curious child asked:" "In a quiet village by the sea," \
  ${GENERATION_DO_SAMPLE_FLAG} \
  ${GENERATION_DEBUG_FLAG}
exit_code=$?
set -e

if (( exit_code != 0 )); then
  echo ""
  echo "[error] Training failed with exit code ${exit_code}. See traceback above for details." >&2
  exit "${exit_code}"
fi

echo ""
echo "========================================="
echo "Training completed!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "========================================="
