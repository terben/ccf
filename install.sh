#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="ccf"
ENV_FILE="ccf.yml"

echo "Setting up conda environment '${ENV_NAME}'... for the ccf package"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda was not found in PATH." >&2
    echo "Please install Miniconda, Anaconda, or Miniforge first." >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Environment '${ENV_NAME}' already exists; updating it..."
    conda env update --name "${ENV_NAME}" --file "${ENV_FILE}" --prune
else
    echo "Creating environment '${ENV_NAME}'..."
    conda env create --file "${ENV_FILE}"
fi

echo "Running test suite..."
conda run --name "${ENV_NAME}" python -m pytest

echo
echo "Installation complete."
echo "Activate the environment with:"
echo "  conda activate ${ENV_NAME}"
