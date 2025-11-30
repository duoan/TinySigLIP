#!/bin/bash
# Download COCO 2017 Caption dataset
#
# This script downloads COCO dataset with default parameters.
# You can override defaults by passing arguments.
#
# Usage:
#   ./download_coco.sh                    # Use all defaults
#   ./download_coco.sh --data-dir /custom/path
#   ./download_coco.sh --split val         # Download only validation split
#   ./download_coco.sh --cleanup           # Clean up zip files after download

set -e  # Exit on error

# Default parameters
DEFAULT_DATA_DIR="${PWD}/data/coco"
DEFAULT_SPLITS=("train" "val" "test")  # Download all splits by default
DEFAULT_ANNOTATIONS_ONLY=false
DEFAULT_CLEANUP=false

# Parse arguments
DATA_DIR="${DEFAULT_DATA_DIR}"
SPLITS=("${DEFAULT_SPLITS[@]}")
ANNOTATIONS_ONLY="${DEFAULT_ANNOTATIONS_ONLY}"
CLEANUP="${DEFAULT_CLEANUP}"

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Download COCO 2017 Caption dataset with default parameters.

Options:
    --data-dir DIR          Data directory (default: ${DEFAULT_DATA_DIR})
    --split SPLIT [SPLIT]   Dataset splits to download: train, val, test (default: all splits)
    --annotations-only      Download only annotations (skip images)
    --cleanup               Remove zip files after extraction to save space
    -h, --help              Show this help message

Examples:
    # Use all defaults (downloads all splits: train, val, test to ./data/coco)
    ./download_coco.sh

    # Download only validation split
    ./download_coco.sh --split val

    # Download to custom directory
    ./download_coco.sh --data-dir /path/to/coco

    # Download only annotations
    ./download_coco.sh --annotations-only

    # Download and cleanup zip files
    ./download_coco.sh --cleanup
EOF
}

# Parse command line arguments
SPLIT_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
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
        --annotations-only)
            ANNOTATIONS_ONLY=true
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
if [[ "$DATA_DIR" = /* ]]; then
    # Already absolute path
    DATA_DIR="$DATA_DIR"
else
    # Relative path - convert to absolute
    DATA_DIR="$(cd "$(dirname "$DATA_DIR")" && pwd)/$(basename "$DATA_DIR")"
fi

# Build Python command
PYTHON_CMD="python download_coco.py --data-dir \"${DATA_DIR}\""

# Add split arguments
if [[ ${#SPLITS[@]} -gt 0 ]]; then
    PYTHON_CMD="${PYTHON_CMD} --split ${SPLITS[*]}"
fi

# Add annotations-only flag
if [[ "${ANNOTATIONS_ONLY}" == "true" ]]; then
    PYTHON_CMD="${PYTHON_CMD} --annotations-only"
fi

# Add cleanup flag
if [[ "${CLEANUP}" == "true" ]]; then
    PYTHON_CMD="${PYTHON_CMD} --cleanup"
fi

# Print configuration
echo "=========================================="
echo "COCO Dataset Download Script"
echo "=========================================="
echo "Data directory: ${DATA_DIR}"
echo "Splits: ${SPLITS[*]}"
echo "Annotations only: ${ANNOTATIONS_ONLY}"
echo "Cleanup zip files: ${CLEANUP}"
echo "=========================================="
echo ""

# Check if Python script exists
if [[ ! -f "download_coco.py" ]]; then
    echo "Error: download_coco.py not found in current directory"
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
    echo "✓ Download completed successfully!"
    echo "=========================================="
    echo ""
    echo "Dataset location:"
    echo "  Images: ${DATA_DIR}/images"
    echo "  Annotations: ${DATA_DIR}/annotations"
    echo ""
    echo "Note: Configuration is already set up in config/config.yaml"
    echo "      with default paths. No manual configuration needed!"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ Download failed!"
    echo "=========================================="
    exit 1
fi
