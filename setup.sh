#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== docling-fast setup ==="

# 1. Require NVIDIA GPU
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. A CUDA GPU is required."
    exit 1
fi

CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)
if [ -z "$CUDA_VERSION" ]; then
    echo "ERROR: Could not detect CUDA version."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "GPU: $GPU_NAME (${GPU_VRAM} MiB), CUDA $CUDA_VERSION"

# 2. Ensure uv is available
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# 3. Find or create a Python environment
# Prefer existing venv (e.g. Vast.AI /venv/main), then .venv, then create one
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "Using active venv: $VIRTUAL_ENV"
elif [ -d ".venv" ]; then
    echo "Activating .venv..."
    source .venv/bin/activate
elif [ -d "/venv/main" ]; then
    echo "Using Vast.AI venv: /venv/main"
    source /venv/main/bin/activate
else
    echo "Creating .venv..."
    uv venv --python 3.12
    source .venv/bin/activate
fi
echo "Python: $(python --version) at $(which python)"

# 4. Check if PyTorch with CUDA is already available
NEED_TORCH=1
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "PyTorch with CUDA already installed, skipping."
    NEED_TORCH=0
fi

if [ "$NEED_TORCH" -eq 1 ]; then
    # Determine PyTorch CUDA index URL
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

    # PyTorch ships cu118, cu121, cu124, cu126, cu128
    if [ "$CUDA_MAJOR" -ge 13 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    elif [ "$CUDA_MAJOR" -eq 12 ]; then
        if [ "$CUDA_MINOR" -ge 8 ]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        elif [ "$CUDA_MINOR" -ge 6 ]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu126"
        elif [ "$CUDA_MINOR" -ge 4 ]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        else
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        fi
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    fi
    echo "Installing PyTorch (${TORCH_INDEX##*/})..."
    uv pip install torch --index-url "$TORCH_INDEX"
fi

# 5. Install docling-fast deps
echo "Installing docling-fast..."
uv pip install -e .

# 6. Verify CUDA works
echo ""
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available after install!'
props = torch.cuda.get_device_properties(0)
print(f'Verified: {props.name}, {props.total_memory / 1024**3:.1f} GB VRAM, PyTorch {torch.__version__}')
print(f'GPUs: {torch.cuda.device_count()}')
"

# 7. Pre-download RapidOCR torch models
echo "Downloading OCR models..."
python -c "
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
RapidOcrModel.download_models(backend='torch', progress=True)
print('Done.')
"

echo ""
echo "=== Ready ==="
echo "  docling-probe           # GPU capabilities"
echo "  docling-fast -i dir/    # Extract PDFs to JSON"
echo "  docling-fast -i dir/ -w 8 -p v100_8x  # Parallel mode"
