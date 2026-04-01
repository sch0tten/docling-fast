"""docling-fast — GPU-accelerated batch PDF extraction with Docling + RapidOCR.

Usage from another project::

    from src.pipeline import build_converter
    from src.extract import extract_one, extract_batch
    from src.config import load_config, AppConfig

    cfg = load_config()                              # or load_config(hardware_profile="blackwell")
    converter = build_converter(cfg)                  # raises RuntimeError if no CUDA
    result = extract_one(Path("doc.pdf"), converter)  # -> dict with "document", "pages", etc.
"""

from .config import AppConfig, load_config
from .extract import extract_batch, extract_one
from .gpu_probe import probe_gpu
from .pipeline import build_converter, configure_docling_settings
