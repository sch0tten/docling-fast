# docling-fast

GPU-accelerated batch PDF extraction using [Docling](https://github.com/docling-project/docling) + [RapidOCR](https://github.com/rapidai/rapidocr) on CUDA. Extracts text, tables, and layout from PDFs — outputs structured JSON.

Built to find the optimal batch sizes for different GPU hardware and run portably on any CUDA machine (local workstation, Vast.AI, cloud GPU).

## Requirements

- NVIDIA GPU with CUDA driver
- Python 3.12+

## Quick Start

```bash
git clone git@github.com:sch0tten/docling-fast.git
cd docling-fast
bash setup.sh           # Detects CUDA, creates .venv, installs deps, downloads models
source .venv/bin/activate
```

`setup.sh` handles everything: detects your CUDA version, picks the right PyTorch wheels, installs Docling and RapidOCR, downloads OCR models, and verifies GPU access.

## CLI Usage

```bash
# Check GPU capabilities and recommended batch sizes
docling-probe

# Extract PDFs to JSON (one JSON file per PDF)
docling-fast -i /path/to/pdfs/ -o output/

# Extract a single PDF
docling-fast -i document.pdf -o output/

# All results in one JSON file
docling-fast -i /path/to/pdfs/ -o output/ --single-file

# Use a hardware-tuned config
docling-fast -i pdfs/ -o output/ -p blackwell

# Benchmark: find optimal batch sizes for your GPU
docling-bench --quick              # 4 configs, 5 docs
docling-bench -p blackwell         # full grid search with profile
docling-bench --max-docs 10        # limit docs per config
```

## Python API

```python
from pathlib import Path
from src import load_config, build_converter, extract_one, extract_batch

# Load config (defaults, or with a hardware profile / overrides)
cfg = load_config()
cfg = load_config(hardware_profile="blackwell")
cfg = load_config(overrides={"pipeline": {"ocr_batch_size": 32}})

# Build converter — raises RuntimeError if no CUDA
converter = build_converter(cfg)

# Extract one PDF
result = extract_one(Path("document.pdf"), converter)
print(result["pages"], result["text_length"], result["tables"])
doc = result["document"]  # full Docling export_to_dict()

# Extract a batch
results = extract_batch([Path("a.pdf"), Path("b.pdf")], converter)
```

### GPU Probe

```python
from src import probe_gpu

info = probe_gpu()
print(info["cuda"]["device_name"])         # "NVIDIA RTX PRO 2000 Blackwell"
print(info["recommended_batch_sizes"])     # {"ocr_batch_size": 32, ...}
```

## How It Works

Docling's pipeline has three GPU-accelerated stages:

1. **Layout detection** — Heron model identifies text blocks, figures, tables
2. **Table structure** — TableFormer extracts rows/columns from detected tables
3. **OCR** — RapidOCR (det + cls + rec) reads text from scanned/image pages

All three run on CUDA via PyTorch. The `torch` backend for RapidOCR shares the same CUDA context as Docling's models — no extra packages needed.

The key tuning knobs are `ocr_batch_size`, `layout_batch_size`, and `table_batch_size` — how many pages are batched per stage. Larger batches use more VRAM but can increase GPU utilization. The benchmark tool grid-searches these to find the sweet spot for your hardware.

## Configuration

TOML files in `config/` with cascading overrides:

```
config/default.toml       <- base defaults
config/rtx_pro_2000.toml  <- tuned for 16 GB VRAM
config/blackwell.toml     <- tuned for 80-192 GB VRAM
```

Override at runtime: `load_config(overrides={"pipeline": {"ocr_batch_size": 64}})`.

Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pipeline.ocr_batch_size` | 4 | Pages per OCR batch |
| `pipeline.layout_batch_size` | 4 | Pages per layout detection batch |
| `pipeline.table_batch_size` | 4 | Pages per table structure batch |
| `pipeline.document_timeout` | 120 | Max seconds per document |
| `pipeline.force_full_page_ocr` | false | OCR entire page vs detected regions |

## Benchmarking

Prepare a sample set of PDFs (the `pdf_catalog.py` tool can help classify OCR-heavy vs text-rich PDFs):

```bash
python -m src.pdf_catalog /path/to/pdfs --samples-dir samples/ --max 50
```

Then run the benchmark:

```bash
docling-bench --quick                    # fast: 4 configs x 5 docs
docling-bench                            # full: all batch size combinations x 50 docs
docling-bench -o results/my_run.json     # custom output path
```

Results are saved as JSON with per-config throughput (pages/sec), VRAM usage, and GPU utilization.

## Project Structure

```
setup.sh             # One-command bootstrap for any CUDA machine
pyproject.toml       # Dependencies and entry points
config/              # TOML configs per hardware profile
src/
  __init__.py        # Public API
  config.py          # Config loader + Pydantic models
  pipeline.py        # DocumentConverter builder
  extract.py         # PDF -> JSON extraction
  benchmark.py       # Batch size grid search
  metrics.py         # Timing + GPU metrics
  gpu_probe.py       # Hardware detection
  pdf_catalog.py     # PDF classification tool
samples/             # Test PDFs (symlinks)
results/             # Benchmark outputs
```

## License

MIT
