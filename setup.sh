#!/bin/bash
# setup.sh — Create conda env for Lie Group RL
#
# Handles:
#   macOS Apple Silicon  → PyTorch with MPS backend
#   macOS Intel          → PyTorch CPU
#   Linux with NVIDIA    → PyTorch with CUDA
#   Linux without GPU    → PyTorch CPU
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# Then:
#   conda activate lie_group_rl
#   python ppo_lie_group.py --state-repr lie_algebra --action-repr lie_algebra

set -e

ENV_NAME="lie_group_rl"

echo "============================================"
echo "  Lie Group RL — Environment Setup"
echo "============================================"
echo ""

# ---- Detect platform ----
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "Platform: $OS $ARCH"

if [[ "$OS" == "Darwin" ]]; then
    if [[ "$ARCH" == "arm64" ]]; then
        PLATFORM="mac_arm"
        echo "  → macOS Apple Silicon (MPS available)"
    else
        PLATFORM="mac_intel"
        echo "  → macOS Intel (CPU only)"
    fi
elif [[ "$OS" == "Linux" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
        if [[ -n "$CUDA_VER" ]]; then
            PLATFORM="linux_cuda"
            echo "  → Linux with CUDA $CUDA_VER"
        else
            PLATFORM="linux_cpu"
            echo "  → Linux (nvidia-smi found but no CUDA version, using CPU)"
        fi
    else
        PLATFORM="linux_cpu"
        echo "  → Linux (no GPU detected, CPU only)"
    fi
else
    PLATFORM="other"
    echo "  → $OS (will attempt generic install)"
fi

echo ""

# ---- Handle existing env ----
if conda info --envs 2>/dev/null | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Recreate from scratch? (y/N): " confirm
    if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
        echo "Removing old environment..."
        conda deactivate 2>/dev/null || true
        conda env remove -n $ENV_NAME -y
    else
        echo ""
        echo "Keeping existing env. To activate:"
        echo "  conda activate $ENV_NAME"
        exit 0
    fi
fi

echo ""

# ---- Install ----
case "$PLATFORM" in

    mac_arm|mac_intel)
        # On macOS, the base environment.yml works as-is.
        # conda's pytorch channel auto-resolves to MPS on arm64.
        echo "Installing via environment.yml (macOS)..."
        conda env create -f environment.yml
        ;;

    linux_cuda)
        # For Linux + CUDA, add the nvidia channel and pin cuda toolkit.
        # Determine CUDA version bucket
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        if [[ "$CUDA_MAJOR" -ge 12 ]]; then
            CUDA_PIN="12.1"
        else
            CUDA_PIN="11.8"
        fi
        echo "Installing with CUDA $CUDA_PIN support..."
        conda create -n $ENV_NAME python=3.10 numpy scipy matplotlib pandas -y
        eval "$(conda shell.bash hook)"
        conda activate $ENV_NAME
        conda install pytorch torchvision pytorch-cuda=$CUDA_PIN -c pytorch -c nvidia -y
        pip install gymnasium>=0.29 mujoco>=3.0 tensorboard>=2.14
        ;;

    linux_cpu|other)
        # CPU-only: use the pip index for a smaller install
        echo "Installing CPU-only variant..."
        conda create -n $ENV_NAME python=3.10 numpy scipy matplotlib pandas -y
        eval "$(conda shell.bash hook)"
        conda activate $ENV_NAME
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        pip install gymnasium>=0.29 mujoco>=3.0 tensorboard>=2.14
        ;;
esac

# ---- Activate and verify ----
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo ""
echo "============================================"
echo "  Verifying installation..."
echo "============================================"
echo ""

python - <<'PYEOF'
import sys
import torch
import numpy as np
import gymnasium
import scipy

print(f"  Python:     {sys.version.split()[0]}")
print(f"  PyTorch:    {torch.__version__}")
print(f"  NumPy:      {np.__version__}")
print(f"  SciPy:      {scipy.__version__}")
print(f"  Gymnasium:  {gymnasium.__version__}")

# Backend detection
if torch.cuda.is_available():
    dev = f"CUDA ({torch.cuda.get_device_name(0)})"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    dev = "MPS (Apple Silicon)"
else:
    dev = "CPU"
print(f"  Device:     {dev}")

try:
    import mujoco
    print(f"  MuJoCo:     {mujoco.__version__}")
except Exception as e:
    print(f"  MuJoCo:     not available ({e})")
PYEOF

echo ""

# ---- Run self-tests ----
echo "============================================"
echo "  Running self-tests..."
echo "============================================"
echo ""

python utils/lie_utils.py
echo ""
python envs/orientation_env.py

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  conda activate $ENV_NAME"
echo ""
echo "  # Quick run (Lie algebra — paper's method):"
echo "  python ppo_lie_group.py --state-repr lie_algebra --action-repr lie_algebra"
echo ""
echo "  # Compare representations:"
echo "  python ppo_lie_group.py --state-repr euler --action-repr euler"
echo "  python ppo_lie_group.py --state-repr quat --action-repr quat"
echo ""
echo "  # Full 9-way comparison:"
echo "  python ppo_lie_group.py --compare"
echo ""
