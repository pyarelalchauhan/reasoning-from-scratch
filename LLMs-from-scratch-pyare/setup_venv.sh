#!/bin/bash

# Setup script for LLMs-from-scratch virtual environment
# Cache directories are set to this project directory, not home

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
CACHE_DIR="$SCRIPT_DIR/.cache"

echo "Setting up virtual environment in: $VENV_DIR"
echo "Cache directory: $CACHE_DIR"

# Create cache directories
mkdir -p "$CACHE_DIR/pip"
mkdir -p "$CACHE_DIR/uv"
mkdir -p "$CACHE_DIR/torch"
mkdir -p "$CACHE_DIR/huggingface"
mkdir -p "$CACHE_DIR/tensorflow"

# Set environment variables for cache locations
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export UV_CACHE_DIR="$CACHE_DIR/uv"
export TORCH_HOME="$CACHE_DIR/torch"
export HF_HOME="$CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$CACHE_DIR/huggingface"
export TENSORFLOW_CACHE_DIR="$CACHE_DIR/tensorflow"

# Create virtual environment with uv
uv venv --python=python3.10 "$VENV_DIR"

# Install dependencies using uv pip (uses venv automatically)
uv pip install --upgrade pip -p "$VENV_DIR"
uv pip install -r "$SCRIPT_DIR/requirements.txt" -p "$VENV_DIR"

# Append cache environment variables to the activate script
cat >> "$VENV_DIR/bin/activate" << EOF

# Local cache directories (added by setup_venv.sh)
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export UV_CACHE_DIR="$CACHE_DIR/uv"
export TORCH_HOME="$CACHE_DIR/torch"
export HF_HOME="$CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$CACHE_DIR/huggingface"
export TENSORFLOW_CACHE_DIR="$CACHE_DIR/tensorflow"
EOF

echo ""
echo "=========================================="
echo "Virtual environment setup complete!"
echo "=========================================="
echo ""
echo "To activate, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Cache directories are automatically set when activated."
echo ""
