#!/bin/bash
# Universal training + data-prep script for TinySigLIP
# 1) Auto-detects machine / GPUs
# 2) Ensures data cache is ready (local: tiny by default, remote: full dataset)
# 3) Starts training (single / multi-GPU / CPU, including macOS Metal)

set -e

# Get the directory where this script is located (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

############################################
# Device + machine detection helpers
############################################

detect_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        # NVIDIA GPUs
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
        GPU_TYPE="NVIDIA"
    elif command -v rocm-smi &> /dev/null; then
        # AMD GPUs (ROCm)
        NUM_GPUS=$(rocm-smi --showid 2>/dev/null | grep -c "GPU" || echo "0")
        GPU_TYPE="AMD"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - check for Metal (Apple Silicon or AMD GPUs)
        if python3 - << 'PY' 2>/dev/null | grep -q "True"; then
import torch
print(getattr(getattr(torch, "backends", None), "mps", None) is not None and torch.backends.mps.is_available())
PY
            NUM_GPUS=1
            GPU_TYPE="Metal"
        else
            NUM_GPUS=0
            GPU_TYPE="CPU"
        fi
    else
        # Fallback: try to detect via Python CUDA
        NUM_GPUS=$(python3 - << 'PY' 2>/dev/null || echo "0"
import torch
print(torch.cuda.device_count())
PY
)
        NUM_GPUS=$(echo "${NUM_GPUS}" | tr -d ' ' || echo "0")
        if [ "${NUM_GPUS}" -gt 0 ] 2>/dev/null; then
            GPU_TYPE="CUDA"
        else
            NUM_GPUS=0
            GPU_TYPE="CPU"
        fi
    fi
}

detect_machine_type() {
    # 最简单清晰的规则：
    # - 如果用户显式设置 REMOTE_MACHINE=true/1 -> 一定视为 remote
    # - 否则：在 Linux 上且 CUDA GPU 数量 > 0 -> remote
    # - 其他情况（macOS / 无 CUDA / 纯 CPU 等） -> local

    if [[ "${REMOTE_MACHINE}" == "true" || "${REMOTE_MACHINE}" == "1" ]]; then
        IS_REMOTE="true"
        return
    fi

    if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "linux"* ]]; then
        if [[ "${GPU_TYPE}" == "NVIDIA" || "${GPU_TYPE}" == "CUDA" ]] && [[ "${NUM_GPUS}" -gt 0 ]]; then
            IS_REMOTE="true"
        else
            IS_REMOTE="false"
        fi
    else
        # macOS / 其它系统默认认为是本地开发机
        IS_REMOTE="false"
    fi
}

############################################
# Data cache detection + preparation
############################################

ensure_data_ready() {
    # Defaults must stay in sync with config/config.yaml and prepare_data.sh
    local CACHE_DIR="${SCRIPT_DIR}/data/coco/cache"
    local TEACHER_MODEL="google/siglip2-base-patch16-224"

    # Sanitize model name for filesystem (same as prepare_data.py)
    local SAFE_MODEL_NAME
    SAFE_MODEL_NAME="$(python3 - << PY
import re
name = "${TEACHER_MODEL}"
safe = re.sub(r"[^\\w\\-_]", "_", name).strip("_")
print(safe)
PY
)"

    local MODEL_CACHE_ROOT="${CACHE_DIR}/${SAFE_MODEL_NAME}"

    echo "Checking TinySigLIP data cache under: ${MODEL_CACHE_ROOT}"

    # Look for any metadata.json under model cache (tiny/medium/large, any split)
    local META_FILE
    META_FILE=$(find "${MODEL_CACHE_ROOT}" -maxdepth 4 -type f -name "metadata.json" 2>/dev/null | head -n 1 || true)

    if [[ -n "${META_FILE}" && -f "${META_FILE}" ]]; then
        echo "✓ Data cache already found: ${META_FILE}"
        return 0
    fi

    echo "⚠ Data cache not found. Running data preparation first..."

    # Decide dataset size: local -> tiny (<=1000), remote -> full dataset
    local PREP_CMD=("bash" "${SCRIPT_DIR}/prepare_data.sh" "--cache-dir" "${CACHE_DIR}" "--teacher-model" "${TEACHER_MODEL}")

    if [[ "${IS_REMOTE}" == "true" ]]; then
        echo "Detected remote machine -> using FULL COCO dataset for data prep."
        # Full dataset: do not pass --max-samples
    else
        echo "Detected local machine -> using TINY COCO subset (max_samples=1000) for data prep."
        PREP_CMD+=("--max-samples" "1000")
    fi

    echo "Running data preparation command:"
    printf '  %q' "${PREP_CMD[@]}"
    echo

    "${PREP_CMD[@]}"

    # Re-check cache after preparation
    META_FILE=$(find "${MODEL_CACHE_ROOT}" -maxdepth 4 -type f -name "metadata.json" 2>/dev/null | head -n 1 || true)
    if [[ -z "${META_FILE}" || ! -f "${META_FILE}" ]]; then
        echo "✗ Data preparation completed but cache metadata.json not found under: ${MODEL_CACHE_ROOT}"
        echo "  Please check the logs of prepare_data.sh for errors."
        exit 1
    fi

    echo "✓ Data cache is ready: ${META_FILE}"
}

############################################
# Training launcher
############################################

launch_training() {
    # Master port for distributed training
    MASTER_PORT=${MASTER_PORT:-29500}

    echo "Detected: ${GPU_TYPE} with ${NUM_GPUS} GPU(s)"

    if [ "${NUM_GPUS}" -gt 1 ] 2>/dev/null; then
        # Multi-GPU training: use torchrun
        echo "Starting multi-GPU training with ${NUM_GPUS} GPUs"
        echo "Master port: ${MASTER_PORT}"

        if command -v torchrun &> /dev/null; then
            torchrun \
                --nproc_per_node="${NUM_GPUS}" \
                --master_port="${MASTER_PORT}" \
                train.py "$@"
        elif python3 -m torch.distributed.launch --help &> /dev/null; then
            # Fallback to torch.distributed.launch for older PyTorch versions
            python3 -m torch.distributed.launch \
                --nproc_per_node="${NUM_GPUS}" \
                --master_port="${MASTER_PORT}" \
                train.py "$@"
        else
            echo "Error: torchrun or torch.distributed.launch not available"
            echo "Falling back to single GPU training"
            python3 train.py "$@"
        fi
    else
        # Single GPU or CPU: run directly
        if [ "${NUM_GPUS}" -eq 1 ] 2>/dev/null; then
            echo "Starting single GPU training"
        else
            echo "Starting CPU/Metal training (no multi-GPU detected)"
        fi
        python3 train.py "$@"
    fi
}

############################################
# Main
############################################

detect_gpus
detect_machine_type

echo "Machine type: $([[ \"${IS_REMOTE}\" == \"true\" ]] && echo 'remote/high-performance' || echo 'local')"

# 1) Ensure data cache is ready (runs prepare_data.sh / prepare_data.py if needed)
ensure_data_ready

# 2) Start training
launch_training "$@"
