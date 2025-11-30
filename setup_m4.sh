#!/bin/bash
# Setup script for GraphiT on M4 Mac
# Usage: bash setup_m4.sh

set -e  # Exit on error

echo "=========================================="
echo "GraphiT Setup for M4 Mac"
echo "=========================================="

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: This script is for macOS only"
    exit 1
fi

# Check for Xcode Command Line Tools
if ! xcode-select -p &> /dev/null; then
    echo "Xcode Command Line Tools not found. Installing..."
    xcode-select --install
    echo "Please run this script again after installation completes"
    exit 1
fi

# Detect Python version
PYTHON_CMD=""
for cmd in python3.11 python3.10 python3.9 python3; do
    if command -v $cmd &> /dev/null; then
        PYTHON_VERSION=$($cmd --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 9 ]; then
            PYTHON_CMD=$cmd
            echo "Using Python: $PYTHON_CMD ($PYTHON_VERSION)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3.9 or higher not found"
    echo "Please install Python 3.11 or higher"
    exit 1
fi

# Determine if using conda or venv
if command -v conda &> /dev/null; then
    echo ""
    echo "Conda detected. Choose installation method:"
    echo "1) Use conda (recommended)"
    echo "2) Use venv"
    read -p "Enter choice [1-2]: " choice

    if [ "$choice" = "1" ]; then
        USE_CONDA=true
    else
        USE_CONDA=false
    fi
else
    echo "Conda not detected. Using venv."
    USE_CONDA=false
fi

# Create environment
if [ "$USE_CONDA" = true ]; then
    echo ""
    echo "Creating conda environment from environment.yml..."
    conda env create -f environment.yml
    echo ""
    echo "To activate the environment, run:"
    echo "  conda activate graphit"
    echo ""
    echo "Then continue with:"
    echo "  pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html"
    echo "  make"
    echo "  mkdir -p cache/pe"
else
    echo ""
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv graphit_env
    source graphit_env/bin/activate

    echo "Upgrading pip..."
    pip install --upgrade pip wheel setuptools

    echo ""
    echo "Installing PyTorch..."
    pip install torch==2.2.0 torchvision torchaudio

    echo ""
    echo "Installing PyTorch Geometric dependencies..."
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
    pip install torch-geometric==2.5.0

    echo ""
    echo "Installing other requirements..."
    pip install -r requirements.txt

    echo ""
    echo "Building C++ extensions..."
    make

    echo ""
    echo "Creating cache directory..."
    mkdir -p cache/pe

    echo ""
    echo "=========================================="
    echo "Installation complete!"
    echo "=========================================="
    echo ""
    echo "To activate the environment in the future, run:"
    echo "  source graphit_env/bin/activate"
    echo ""
    echo "To verify installation:"
    echo "  cd experiments"
    echo '  python -c "import torch; import torch_geometric; print(\"PyTorch:\", torch.__version__); print(\"PyG:\", torch_geometric.__version__)"'
    echo ""
    echo "To run a test:"
    echo "  cd experiments"
    echo "  python run_transformer_cv.py --dataset NCI1 --fold-idx 1 --pos-enc diffusion --beta 1.0"
fi
