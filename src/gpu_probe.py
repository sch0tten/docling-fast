"""GPU detection and capability reporting."""

import json
import sys


def probe_gpu() -> dict:
    """Detect GPU hardware and recommend Docling batch sizes.

    Returns a dict with gpu info, cuda status, and recommended_batch_sizes.
    Exits with error if no CUDA GPU is available.
    """
    result = {
        "gpu": None,
        "cuda": {"available": False},
        "recommended_batch_sizes": {},
    }
    vram_gb = 0

    # Physical GPU info via NVML
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        result["gpu"] = {
            "name": pynvml.nvmlDeviceGetName(handle),
            "vram_total_mb": mem_info.total // (1024 * 1024),
            "vram_free_mb": mem_info.free // (1024 * 1024),
            "driver": pynvml.nvmlSystemGetDriverVersion(),
            "device_count": pynvml.nvmlDeviceGetCount(),
        }
        vram_gb = mem_info.total / (1024**3)
        pynvml.nvmlShutdown()
    except Exception as e:
        result["gpu"] = {"error": str(e)}

    # PyTorch CUDA
    try:
        import torch

        result["cuda"] = {
            "available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            result["cuda"].update({
                "device_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "vram_gb": round(props.total_memory / (1024**3), 1),
                "cuda_version": torch.version.cuda,
            })
            if vram_gb == 0:
                vram_gb = props.total_memory / (1024**3)
    except ImportError:
        result["cuda"] = {"available": False, "error": "torch not installed"}

    # Recommended batch sizes based on VRAM
    if vram_gb <= 8:
        sizes = {"ocr_batch_size": 8, "layout_batch_size": 8, "table_batch_size": 8}
    elif vram_gb <= 16:
        sizes = {"ocr_batch_size": 32, "layout_batch_size": 32, "table_batch_size": 16}
    elif vram_gb <= 48:
        sizes = {"ocr_batch_size": 48, "layout_batch_size": 48, "table_batch_size": 32}
    else:
        sizes = {"ocr_batch_size": 64, "layout_batch_size": 64, "table_batch_size": 64}
    result["recommended_batch_sizes"] = sizes

    return result


def main():
    result = probe_gpu()
    print(json.dumps(result, indent=2))
    if not result["cuda"]["available"]:
        print("\nERROR: No CUDA GPU available. docling-fast requires a GPU.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
