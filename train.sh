#!/bin/bash
# Universal training script for TinySigLIP
# Automatically handles single GPU, multi-GPU, and CPU (including MacBook) training

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect available GPUs
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
    if python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
        NUM_GPUS=1
        GPU_TYPE="Metal"
    else
        NUM_GPUS=0
        GPU_TYPE="CPU"
    fi
else
    # Fallback: try to detect via Python
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    if [ "$NUM_GPUS" -gt 0 ]; then
        GPU_TYPE="CUDA"
    else
        NUM_GPUS=0
        GPU_TYPE="CPU"
    fi
fi

# Master port for distributed training
MASTER_PORT=${MASTER_PORT:-29500}

echo "Detected: $GPU_TYPE with $NUM_GPUS GPU(s)"

# Determine training mode
if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training: use torchrun
    echo "Starting multi-GPU training with $NUM_GPUS GPUs"
    echo "Master port: $MASTER_PORT"
    
    if command -v torchrun &> /dev/null; then
        torchrun \
            --nproc_per_node=$NUM_GPUS \
            --master_port=$MASTER_PORT \
            train.py "$@"
    elif python3 -m torch.distributed.launch --help &> /dev/null; then
        # Fallback to torch.distributed.launch for older PyTorch versions
        python3 -m torch.distributed.launch \
            --nproc_per_node=$NUM_GPUS \
            --master_port=$MASTER_PORT \
            train.py "$@"
    else
        echo "Error: torchrun or torch.distributed.launch not available"
        echo "Falling back to single GPU training"
        python3 train.py "$@"
    fi
else
    # Single GPU or CPU: run directly
    if [ "$NUM_GPUS" -eq 1 ]; then
        echo "Starting single GPU training"
    else
        echo "Starting CPU training (no GPU detected)"
    fi
    python3 train.py "$@"
fi
