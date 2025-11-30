#!/bin/bash
# Helper script to run Python files using the project's virtual environment

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "${SCRIPT_DIR}/.venv/bin/activate"

# Run the Python file(s) passed as arguments
if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh <python_file> [args...]"
    echo "Example: ./run.sh main.py"
    echo "Example: ./run.sh tinysiglip/model.py"
    exit 1
fi

# Execute Python with the provided arguments
exec "${SCRIPT_DIR}/.venv/bin/python" "$@"
