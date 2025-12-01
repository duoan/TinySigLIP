#!/bin/bash
# Unified data preparation script for TinySigLIP
#
# This script handles all data preparation steps:
# 1. Download COCO 2017 dataset (images + annotations)
# 2. Extract/unzip the downloaded files
# 3. Extract and cache teacher model embeddings
#
# Usage:
#   ./prepare_data.sh                    # Use all defaults (auto-detects machine type)
#   ./prepare_data.sh --cache-dir /custom/path/cache
#   ./prepare_data.sh --max-samples 1000  # Small dataset for local dev (1K samples)
#   ./prepare_data.sh --skip-download    # Skip download if data exists
#   ./prepare_data.sh --skip-embeddings  # Skip embedding extraction
#
# Auto-detection:
#   The script automatically detects machine type and adjusts batch_size and num_workers:
#   - Local machines: batch_size=32, num_workers=16 (I/O optimized, scales with CPU cores)
#   - Remote/high-performance machines: batch_size=128, num_workers=64-128 (I/O optimized)
#   - num_workers is optimized for I/O-intensive image downloads (can use many concurrent workers)
#   - Set REMOTE_MACHINE=true to force high-performance mode
#   - You can still override with --batch-size and --num-workers flags

set -e  # Exit on error

# Function to detect machine type and set optimal defaults
detect_machine_type() {
    local cpu_cores
    local total_memory_gb
    local gpu_count=0
    local is_remote=false

    # Detect CPU cores
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        cpu_cores=$(sysctl -n hw.ncpu 2>/dev/null || echo "4")
        total_memory_gb=$(($(sysctl -n hw.memsize 2>/dev/null || echo "8589934592") / 1024 / 1024 / 1024))
    else
        # Linux
        cpu_cores=$(nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo 2>/dev/null || echo "4")
        total_memory_gb=$(($(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo "8388608") / 1024 / 1024))
    fi

    # Detect GPU count (if nvidia-smi is available)
    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")
    fi

    # Check if this is a remote/high-performance machine
    # Check environment variable first
    if [[ "${REMOTE_MACHINE}" == "true" ]] || [[ "${REMOTE_MACHINE}" == "1" ]]; then
        is_remote=true
    # Check hostname patterns (common remote machine patterns)
    elif [[ -n "${HOSTNAME}" ]] && [[ "${HOSTNAME}" =~ ^(gpu|compute|node|server|remote|cluster|a100|h100|v100) ]]; then
        is_remote=true
    # Check if machine has high-end specs
    elif [[ $cpu_cores -ge 32 ]] || [[ $total_memory_gb -ge 64 ]] || [[ $gpu_count -ge 2 ]]; then
        is_remote=true
    fi

    # Set defaults based on machine type
    if [[ "$is_remote" == "true" ]]; then
        # High-performance remote machine: use larger values
        DEFAULT_BATCH_SIZE=128
        # For I/O-intensive tasks (image download), use more workers
        # I/O operations spend most time waiting, so we can use many concurrent workers
        if [[ $cpu_cores -ge 64 ]]; then
            DEFAULT_NUM_WORKERS=128
        elif [[ $cpu_cores -ge 32 ]]; then
            DEFAULT_NUM_WORKERS=64
        elif [[ $cpu_cores -ge 16 ]]; then
            DEFAULT_NUM_WORKERS=32
        else
            DEFAULT_NUM_WORKERS=$((cpu_cores * 2))
        fi
        echo "Detected high-performance machine:"
        echo "  CPU cores: ${cpu_cores}"
        echo "  Memory: ${total_memory_gb} GB"
        echo "  GPUs: ${gpu_count}"
        echo "  Using batch_size=${DEFAULT_BATCH_SIZE}, num_workers=${DEFAULT_NUM_WORKERS} (I/O optimized)"
    else
        # Local machine: use conservative values
        DEFAULT_BATCH_SIZE=32
        # For local machines, still use more workers for I/O but cap it reasonably
        DEFAULT_NUM_WORKERS=$((cpu_cores > 16 ? 16 : (cpu_cores > 0 ? cpu_cores * 2 : 8)))
        echo "Detected local machine:"
        echo "  CPU cores: ${cpu_cores}"
        echo "  Memory: ${total_memory_gb} GB"
        echo "  GPUs: ${gpu_count}"
        echo "  Using batch_size=${DEFAULT_BATCH_SIZE}, num_workers=${DEFAULT_NUM_WORKERS} (I/O optimized)"
    fi
    echo ""
}

# Default parameters (will be auto-adjusted by detect_machine_type)
DEFAULT_CACHE_DIR="${PWD}/data/coco/cache"
DEFAULT_SPLITS=("train" "val")
DEFAULT_TEACHER_MODEL="google/siglip2-base-patch16-224"
# These will be set by detect_machine_type()
DEFAULT_BATCH_SIZE=32
DEFAULT_NUM_WORKERS=8

# Auto-detect machine type and adjust defaults
detect_machine_type

# Parse arguments
CACHE_DIR="${DEFAULT_CACHE_DIR}"
SPLITS=("${DEFAULT_SPLITS[@]}")
TEACHER_MODEL="${DEFAULT_TEACHER_MODEL}"
BATCH_SIZE="${DEFAULT_BATCH_SIZE}"
NUM_WORKERS="${DEFAULT_NUM_WORKERS}"
MAX_SAMPLES=""
SKIP_DOWNLOAD=false
SKIP_EMBEDDINGS=false
CLEANUP=false

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Unified data preparation script for TinySigLIP.

Options:
    --cache-dir DIR         Cache directory (default: ${DEFAULT_CACHE_DIR})
    --split SPLIT [SPLIT]   Dataset splits to download: train, val, test (default: train val)
    --teacher-model MODEL   Teacher model name (default: ${DEFAULT_TEACHER_MODEL})
    --batch-size SIZE       Batch size for embedding extraction (auto-detected, can override)
    --max-samples N         Maximum number of samples to use (for small dataset, e.g., 1000 for local dev)
    --num-workers N         Number of parallel workers for image download (auto-detected, can override)
    --skip-download         Skip downloading COCO dataset
    --skip-embeddings       Skip extracting teacher embeddings
    --cleanup               Remove zip files after extraction
    -h, --help              Show this help message

Auto-detection:
    The script automatically detects machine type and sets optimal defaults:
    - Local machines: batch_size=32, num_workers=16 (I/O optimized, scales with CPU cores)
    - Remote/high-performance machines: batch_size=128, num_workers=64-128 (I/O optimized)
    - Set REMOTE_MACHINE=true environment variable to force high-performance mode
    - Detection is based on CPU cores, memory, GPU count, and hostname patterns
    - num_workers is optimized for I/O-intensive image downloads (can use many concurrent workers)

Examples:
    # Use all defaults (downloads train/val + extracts embeddings)
    ./prepare_data.sh

    # Small dataset for local development (1K samples)
    ./prepare_data.sh --max-samples 1000

    # Skip download if data already exists
    ./prepare_data.sh --skip-download

    # Only download, skip embedding extraction
    ./prepare_data.sh --skip-embeddings

    # Custom cache directory
    ./prepare_data.sh --cache-dir /path/to/coco/cache

    # Use more parallel workers for faster download
    ./prepare_data.sh --max-samples 5000 --num-workers 16
EOF
}

# Parse command line arguments
SPLIT_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --cache-dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --split)
            SPLIT_ARGS=()
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                SPLIT_ARGS+=("$1")
                shift
            done
            if [[ ${#SPLIT_ARGS[@]} -gt 0 ]]; then
                SPLITS=("${SPLIT_ARGS[@]}")
            fi
            ;;
        --teacher-model)
            TEACHER_MODEL="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-embeddings)
            SKIP_EMBEDDINGS=true
            shift
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Convert to absolute path
if [[ "$CACHE_DIR" = /* ]]; then
    # Already absolute path
    CACHE_DIR="$CACHE_DIR"
else
    # Relative path - convert to absolute
    CACHE_DIR="$(cd "$(dirname "$CACHE_DIR")" && pwd)/$(basename "$CACHE_DIR")"
fi

# Build Python command
PYTHON_CMD="python prepare_data.py --cache-dir \"${CACHE_DIR}\""

# Add split argument (Python script only accepts one split)
if [[ ${#SPLITS[@]} -gt 0 ]]; then
    # Use first split only (Python script only accepts one split at a time)
    PYTHON_CMD="${PYTHON_CMD} --split ${SPLITS[0]}"
    if [[ ${#SPLITS[@]} -gt 1 ]]; then
        echo "Warning: Python script only accepts one split at a time. Using first split: ${SPLITS[0]}"
        echo "         To process multiple splits, run the script multiple times."
    fi
fi

# Add teacher model
PYTHON_CMD="${PYTHON_CMD} --teacher-model \"${TEACHER_MODEL}\""

# Add batch size
PYTHON_CMD="${PYTHON_CMD} --batch-size ${BATCH_SIZE}"

# Add max samples if specified
if [[ -n "${MAX_SAMPLES}" ]]; then
    PYTHON_CMD="${PYTHON_CMD} --max-samples ${MAX_SAMPLES}"
fi

# Add num workers
PYTHON_CMD="${PYTHON_CMD} --num-workers ${NUM_WORKERS}"

# Add flags
if [[ "${SKIP_DOWNLOAD}" == "true" ]]; then
    PYTHON_CMD="${PYTHON_CMD} --skip-download"
fi

if [[ "${SKIP_EMBEDDINGS}" == "true" ]]; then
    PYTHON_CMD="${PYTHON_CMD} --skip-embeddings"
fi

if [[ "${CLEANUP}" == "true" ]]; then
    PYTHON_CMD="${PYTHON_CMD} --cleanup"
fi

# Print configuration
echo "=========================================="
echo "TinySigLIP Data Preparation Script"
echo "=========================================="
echo "Cache directory: ${CACHE_DIR}"
echo "Splits: ${SPLITS[*]}"
echo "Teacher model: ${TEACHER_MODEL}"
echo "Batch size: ${BATCH_SIZE}"
if [[ -n "${MAX_SAMPLES}" ]]; then
    echo "Max samples: ${MAX_SAMPLES} (small dataset for local dev)"
else
    echo "Max samples: Full dataset"
fi
echo "Num workers: ${NUM_WORKERS}"
echo "Skip download: ${SKIP_DOWNLOAD}"
echo "Skip embeddings: ${SKIP_EMBEDDINGS}"
echo "Cleanup: ${CLEANUP}"
echo "=========================================="
echo ""

# Check if Python script exists
if [[ ! -f "prepare_data.py" ]]; then
    echo "Error: prepare_data.py not found in current directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.7+"
    exit 1
fi

# Use python3 if python is not available
if command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    PYTHON_CMD=$(echo "$PYTHON_CMD" | sed 's/python /python3 /')
fi

# Execute Python script
echo "Running: ${PYTHON_CMD}"
echo ""
eval "${PYTHON_CMD}"

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo "=========================================="
    echo "✓ Data preparation completed successfully!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Check config/config.yaml (paths should be set automatically)"
    echo "  2. Start training: python train.py"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ Data preparation failed!"
    echo "=========================================="
    exit 1
fi
