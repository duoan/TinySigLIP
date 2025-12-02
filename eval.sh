#!/bin/bash
# End-to-end evaluation script for TinySigLIP on COCO val
# 1) Prepare COCO val data + teacher embeddings
# 2) List local checkpoints and let user choose (with option to pick the most recent)

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${SCRIPT_DIR}/data/coco"
CACHE_DIR="${DATA_DIR}/cache"
TEACHER_MODEL="google/siglip2-base-patch16-224"
SPLIT="val"

echo "==============================================="
echo " TinySigLIP COCO Val Evaluation"
echo "==============================================="
echo "Project dir  : ${SCRIPT_DIR}"
echo "Data dir     : ${DATA_DIR}"
echo "Cache dir    : ${CACHE_DIR}"
echo "Teacher model: ${TEACHER_MODEL}"
echo "Split        : ${SPLIT}"
echo "==============================================="
echo

########################################
# Step 1: Prepare COCO val data + embeddings
########################################

echo ">>> Step 1: Preparing COCO '${SPLIT}' data and teacher embeddings"
echo

# Ensure base directories exist
mkdir -p "${DATA_DIR}"
mkdir -p "${CACHE_DIR}"

python3 "${SCRIPT_DIR}/prepare_data.py" \
    --split "${SPLIT}" \
    --teacher-model "${TEACHER_MODEL}" \
    --cache-dir "${CACHE_DIR}"

echo
echo ">>> Data & embeddings preparation done."
echo

########################################
# Step 2: Select checkpoint for evaluation
########################################

echo ">>> Step 2: Selecting checkpoint to evaluate"

# Find all checkpoint.pt files under outputs (sorted by modification time, newest first)
mapfile -t CHECKPOINTS < <(find "${SCRIPT_DIR}/outputs" -maxdepth 3 -type f -name "checkpoint.pt" -print0 2>/dev/null | xargs -0 ls -t 2>/dev/null || true)

if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
    echo "No checkpoint.pt files found under '${SCRIPT_DIR}/outputs'."
    echo "Please run training first (e.g., ./train.sh) and then re-run this script."
    exit 1
fi

echo
echo "Found the following checkpoints:"
echo
echo "  0) Use most recent checkpoint:"
echo "     ${CHECKPOINTS[0]}"
echo

idx=1
for ckpt in "${CHECKPOINTS[@]}"; do
    echo "  ${idx}) ${ckpt}"
    idx=$((idx + 1))
done

echo
read -rp "Select checkpoint index [0-$((${#CHECKPOINTS[@]}))] (default: 0): " CHOICE

# Default to 0 if empty
if [ -z "${CHOICE}" ]; then
    CHOICE=0
fi

if ! [[ "${CHOICE}" =~ ^[0-9]+$ ]]; then
    echo "Invalid choice: ${CHOICE}"
    exit 1
fi

if [ "${CHOICE}" -eq 0 ]; then
    SELECTED_CKPT="${CHECKPOINTS[0]}"
else
    INDEX=$((CHOICE - 1))
    if [ "${INDEX}" -lt 0 ] || [ "${INDEX}" -ge "${#CHECKPOINTS[@]}" ]; then
        echo "Choice out of range."
        exit 1
    fi
    SELECTED_CKPT="${CHECKPOINTS[INDEX]}"
fi

echo
echo ">>> Selected checkpoint:"
echo "    ${SELECTED_CKPT}"
echo

########################################
# Step 3: Run evaluation
########################################

echo ">>> Step 3: Running evaluation on COCO '${SPLIT}' split"
echo

python3 "${SCRIPT_DIR}/eval.py" \
    --resume "${SELECTED_CKPT}" \
    --split "${SPLIT}"

echo
echo "==============================================="
echo " Evaluation finished."
echo "==============================================="
