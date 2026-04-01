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

# 3. Create .venv
if [ ! -d ".venv" ]; then
    echo "Creating .venv..."
    uv venv --python 3.12
fi
source .venv/bin/activate

# 4. Determine PyTorch CUDA index URL
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
echo "PyTorch index: ${TORCH_INDEX##*/}"

# 5. Install PyTorch with CUDA, then project deps
echo "Installing PyTorch..."
uv pip install torch --index-url "$TORCH_INDEX"

echo "Installing docling-fast..."
uv pip install -e .

# 6. Verify CUDA works
echo ""
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available after install!'
props = torch.cuda.get_device_properties(0)
print(f'Verified: {props.name}, {props.total_memory / 1024**3:.1f} GB VRAM, PyTorch {torch.__version__}')
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
echo "  source .venv/bin/activate"
echo "  docling-probe           # GPU capabilities"
echo "  docling-fast -i dir/    # Extract PDFs to JSON"
echo "  docling-bench --quick   # Benchmark batch sizes"
