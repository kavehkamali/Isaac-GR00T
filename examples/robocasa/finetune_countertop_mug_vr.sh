#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <converted_lerobot_dataset_path> [output_dir] [extra_launch_finetune_args...]"
    exit 1
fi

DATASET_PATH="$1"
OUTPUT_DIR="${2:-/tmp/robocasa_vr_countertop_mug_finetune}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.6-3B}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-examples/robocasa/robocasa_vr_modality_config.py}"

MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "Dataset path not found: ${DATASET_PATH}"
    exit 1
fi

if [[ ! -f "${MODALITY_CONFIG_PATH}" ]]; then
    echo "Modality config not found: ${MODALITY_CONFIG_PATH}"
    exit 1
fi

WANDB_ARGS=()
if [[ "${USE_WANDB:-0}" == "1" ]]; then
    WANDB_ARGS+=(--use_wandb)
fi

EXTRA_ARGS=()
if [[ $# -gt 2 ]]; then
    EXTRA_ARGS=("${@:3}")
fi

COMMON_ARGS=(
    gr00t/experiment/launch_finetune.py
    --base_model_path "${BASE_MODEL_PATH}"
    --dataset_path "${DATASET_PATH}"
    --embodiment_tag NEW_EMBODIMENT
    --modality_config_path "${MODALITY_CONFIG_PATH}"
    --num_gpus "${NUM_GPUS}"
    --output_dir "${OUTPUT_DIR}"
    --save_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --max_steps "${MAX_STEPS}"
    --warmup_ratio "${WARMUP_RATIO}"
    --weight_decay "${WEIGHT_DECAY}"
    --learning_rate "${LEARNING_RATE}"
    --global_batch_size "${GLOBAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
)

if [[ "${NUM_GPUS}" -gt 1 ]]; then
    torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
        "${COMMON_ARGS[@]}" \
        "${WANDB_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
else
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python \
        "${COMMON_ARGS[@]}" \
        "${WANDB_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
fi
