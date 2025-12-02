#!/bin/bash
# End-to-end evaluation script for TinySigLIP on COCO val
# 1) Prepare COCO val data + teacher embeddings (skip if cache exists)
# 2) List local checkpoints and let user choose (with option to pick the most recent)
#
# Usage:
#   ./eval.sh                    # Normal evaluation (skip data prep if cache exists)
#   ./eval.sh --force-prepare     # Force re-preparation of data and embeddings

set -e

# Parse command line arguments
FORCE_PREPARE=0
if [ "$1" = "--force-prepare" ] || [ "$1" = "-f" ]; then
    FORCE_PREPARE=1
    echo ">>> Force prepare mode: will regenerate data and embeddings even if cache exists"
    echo
fi

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
# Step 1: Prepare COCO val data + embeddings (skip if already exists)
########################################

echo ">>> Step 1: Checking if COCO '${SPLIT}' data and teacher embeddings exist"
echo

# Ensure base directories exist
mkdir -p "${DATA_DIR}"
mkdir -p "${CACHE_DIR}"

# Sanitize teacher model name for filesystem (same logic as prepare_data.py)
SAFE_MODEL_NAME=$(echo "${TEACHER_MODEL}" | sed 's/[^a-zA-Z0-9_-]/_/g' | sed 's/^_\|_$//g')

# Check for cache in different dataset sizes (tiny, medium, large)
# Try large first (default), then medium, then tiny
CACHE_FOUND=0
for DATASET_SIZE in "large" "medium" "tiny"; do
    CACHE_PATH="${CACHE_DIR}/${SAFE_MODEL_NAME}/${DATASET_SIZE}/${SPLIT}"
    METADATA_FILE="${CACHE_PATH}/metadata.json"

    if [ -f "${METADATA_FILE}" ]; then
        echo "✓ Found existing cache at: ${CACHE_PATH}"
        echo "  Metadata file: ${METADATA_FILE}"

        # Check if there are batch files (indicating embeddings are complete)
        BATCH_COUNT=$(find "${CACHE_PATH}" -maxdepth 1 -name "batch_*.pkl" 2>/dev/null | wc -l | tr -d ' ')
        if [ "${BATCH_COUNT}" -gt 0 ]; then
            echo "  Found ${BATCH_COUNT} batch file(s) - embeddings appear complete"
            CACHE_FOUND=1
            break
        else
            echo "  ⚠ Warning: Metadata exists but no batch files found"
        fi
    fi
done

if [ "${CACHE_FOUND}" -eq 0 ] || [ "${FORCE_PREPARE}" -eq 1 ]; then
    if [ "${FORCE_PREPARE}" -eq 1 ]; then
        echo "⚠ Force prepare mode: Regenerating COCO '${SPLIT}' data and teacher embeddings..."
    else
        echo "⚠ Cache not found. Preparing COCO '${SPLIT}' data and teacher embeddings..."
    fi
    echo "  This may take a while, especially for embedding extraction."
    echo

    python3 "${SCRIPT_DIR}/prepare_data.py" \
        --split "${SPLIT}" \
        --teacher-model "${TEACHER_MODEL}" \
        --cache-dir "${CACHE_DIR}"

    echo
    echo ">>> Data & embeddings preparation done."
else
    echo
    echo ">>> Skipping data preparation (cache already exists)"
    echo "   Use --force-prepare to regenerate data and embeddings"
fi
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
