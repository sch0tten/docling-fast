# CLAUDE.md — docling-fast

## What This Is

Reusable component for GPU-accelerated PDF extraction. Wraps Docling + RapidOCR with the `torch` backend on CUDA. All processing runs on GPU — no CPU fallback. Outputs JSON.

Designed to be portable across any CUDA-capable machine (local dev, Vast.AI, cloud GPU instances).

## Setup

```bash
bash setup.sh             # Detects CUDA, creates .venv, installs everything, verifies GPU
source .venv/bin/activate
```

Requires: NVIDIA GPU with CUDA driver, Python 3.12+.

## CLI Tools

```bash
docling-probe                        # Print GPU info + recommended batch sizes as JSON
docling-fast -i some/dir/ -o out/    # Extract all PDFs to JSON (one .json per PDF)
docling-fast -i file.pdf --single-file  # All results into one JSON file
docling-bench --quick                # Quick benchmark (4 configs, 5 docs)
docling-bench -p blackwell           # Full benchmark with hardware profile
```

## Use as a Library

From another project that has this component's `.venv` on its path, or after `pip install -e /path/to/docling-fast`:

```python
from pathlib import Path
from src import load_config, build_converter, extract_one, extract_batch

# 1. Load config (uses config/default.toml, override with hardware profile)
cfg = load_config()                                    # defaults
cfg = load_config(hardware_profile="blackwell")        # or a profile name
cfg = load_config(overrides={"pipeline": {"ocr_batch_size": 32}})  # runtime override

# 2. Build GPU converter (raises RuntimeError if no CUDA)
converter = build_converter(cfg)

# 3a. Extract one PDF
result = extract_one(Path("document.pdf"), converter)
# result = {"source": "document.pdf", "pages": 12, "extraction_time_s": 3.2,
#           "text_length": 45000, "tables": 3, "success": True, "document": {...}}

# 3b. Extract many PDFs
results = extract_batch([Path("a.pdf"), Path("b.pdf")], converter)

# 4. Access the raw Docling dict
doc = result["document"]  # full export_to_dict() output
texts = doc["texts"]      # list of text items with "text", "label", "prov" keys
tables = doc["tables"]    # list of table structures
pages = doc["pages"]      # page-level info
```

### Programmatic GPU probe

```python
from src import probe_gpu

info = probe_gpu()
# info["cuda"]["available"], info["cuda"]["vram_gb"], info["recommended_batch_sizes"]
```

### Programmatic config

```python
from src.config import AppConfig, PipelineConfig

cfg = AppConfig(pipeline=PipelineConfig(ocr_batch_size=32, layout_batch_size=32))
```

## GPU Pipeline

Three model stages, all on CUDA:
1. **Layout detection** — Docling's Heron model, `AcceleratorOptions(device="cuda:0")`
2. **Table structure** — Docling's TableFormer, same AcceleratorOptions
3. **OCR (det/cls/rec)** — RapidOCR with `backend="torch"`, `use_cuda=True`

## Configuration

TOML files in `config/`. Cascading: `default.toml` -> hardware profile -> runtime overrides.

Key knobs:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ocr_batch_size` | 4 | Pages per OCR batch (increase for more VRAM, higher GPU util) |
| `layout_batch_size` | 4 | Pages per layout detection batch |
| `table_batch_size` | 4 | Pages per table structure batch |
| `document_timeout` | 120 | Seconds per document (0 = no timeout) |
| `force_full_page_ocr` | false | OCR entire page vs only detected text regions |

Hardware profiles: `config/rtx_pro_2000.toml`, `config/blackwell.toml`.

## Project Structure

```
src/
  __init__.py      # Public API: load_config, build_converter, extract_one, extract_batch, probe_gpu
  config.py        # TOML config loader, Pydantic models
  pipeline.py      # DocumentConverter builder with GPU config
  extract.py       # PDF -> JSON extraction (single + batch)
  benchmark.py     # Grid search over batch sizes
  metrics.py       # Timing, throughput, VRAM tracking
  gpu_probe.py     # GPU detection and batch size recommendations
  pdf_catalog.py   # Scan dirs, classify OCR-heavy PDFs, build sample sets
config/            # TOML config files per hardware profile
samples/           # PDF symlinks for benchmarking
results/           # Benchmark output JSONs
```

## Rules

- GPU only. Never CPU. Exits immediately if CUDA is unavailable.
- Own `.venv` — do not share with parent projects.
- Sample PDFs in `samples/` are symlinks — never modify the targets.
