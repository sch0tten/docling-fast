"""Multi-GPU multi-worker pool for parallel PDF extraction.

Architecture: N GPUs x M workers per GPU. Each worker is a separate process
with CUDA_VISIBLE_DEVICES pinned to one GPU. Workers pull PDFs from a shared
queue and write JSON results to the output directory.

The GPU stays saturated because while one worker does CPU-bound PDF parsing,
another worker's GPU kernels execute.
"""

import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

from .config import AppConfig

_log = logging.getLogger(__name__)


def _detect_gpu_count() -> int:
    """Detect number of CUDA GPUs available."""
    import torch
    return torch.cuda.device_count()


def _worker_process(
    worker_id: int,
    gpu_id: int,
    pdf_queue: mp.Queue,
    result_queue: mp.Queue,
    output_dir: str,
    cfg_dict: dict,
):
    """Worker process: builds its own converter, processes PDFs from queue.

    Each worker pins to a single GPU via CUDA_VISIBLE_DEVICES so that
    device="cuda:0" in the config maps to the assigned physical GPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import and configure inside the subprocess (fresh Python state)
    from .config import AppConfig
    from .pipeline import build_converter, configure_docling_settings
    from .extract import extract_one

    cfg = AppConfig(**cfg_dict)
    configure_docling_settings(cfg)

    logging.basicConfig(level=logging.WARNING, format=f"[w{worker_id}:gpu{gpu_id}] %(message)s")

    try:
        converter = build_converter(cfg)
    except Exception as e:
        _log.error(f"Worker {worker_id} failed to build converter: {e}")
        result_queue.put({"worker_id": worker_id, "error": str(e), "docs_processed": 0})
        return

    out = Path(output_dir)
    docs_processed = 0
    total_pages = 0
    total_time = 0.0

    while True:
        item = pdf_queue.get()
        if item is None:  # poison pill
            break

        pdf_path = Path(item)
        result = extract_one(pdf_path, converter)

        # Write JSON per document
        out_file = out / f"{pdf_path.stem}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        docs_processed += 1
        total_pages += result.get("pages", 0)
        total_time += result.get("extraction_time_s", 0)

        status = "OK" if result["success"] else "FAIL"
        print(f"  [w{worker_id}:gpu{gpu_id}] {pdf_path.name} {status} "
              f"({result.get('extraction_time_s', 0):.1f}s, {result.get('pages', 0)}p)")

    result_queue.put({
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "docs_processed": docs_processed,
        "total_pages": total_pages,
        "total_time_s": round(total_time, 2),
    })


def run_parallel(
    pdf_paths: list[Path],
    cfg: AppConfig,
    output_dir: Path,
) -> dict:
    """Run parallel extraction across multiple GPUs and workers.

    Args:
        pdf_paths: List of PDF files to process
        cfg: Application config
        output_dir: Directory to write JSON results

    Returns:
        Summary dict with throughput metrics and per-worker stats.
    """
    num_gpus = cfg.workers.num_gpus or _detect_gpu_count()
    workers_per_gpu = cfg.workers.workers_per_gpu
    total_workers = num_gpus * workers_per_gpu

    print(f"Launching {total_workers} workers ({workers_per_gpu} per GPU x {num_gpus} GPUs)")
    print(f"Processing {len(pdf_paths)} PDFs")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use spawn — required for CUDA in subprocesses
    ctx = mp.get_context("spawn")
    pdf_queue = ctx.Queue()
    result_queue = ctx.Queue()

    # Enqueue all PDFs
    for p in pdf_paths:
        pdf_queue.put(str(p))

    # Poison pills (one per worker)
    for _ in range(total_workers):
        pdf_queue.put(None)

    # Serialize config to dict (Pydantic model can't be pickled across spawn)
    cfg_dict = cfg.model_dump()

    # Start workers
    wall_start = time.perf_counter()
    processes = []
    for gpu_id in range(num_gpus):
        for w in range(workers_per_gpu):
            worker_id = gpu_id * workers_per_gpu + w
            p = ctx.Process(
                target=_worker_process,
                args=(worker_id, gpu_id, pdf_queue, result_queue, str(output_dir), cfg_dict),
            )
            p.start()
            processes.append(p)

    # Wait for all workers to finish
    for p in processes:
        p.join()

    wall_time = time.perf_counter() - wall_start

    # Collect worker summaries
    worker_stats = []
    while not result_queue.empty():
        worker_stats.append(result_queue.get())

    total_docs = sum(w.get("docs_processed", 0) for w in worker_stats)
    total_pages = sum(w.get("total_pages", 0) for w in worker_stats)

    summary = {
        "num_gpus": num_gpus,
        "workers_per_gpu": workers_per_gpu,
        "total_workers": total_workers,
        "total_docs": total_docs,
        "total_pages": total_pages,
        "wall_time_s": round(wall_time, 2),
        "pages_per_second": round(total_pages / wall_time, 2) if wall_time > 0 else 0,
        "docs_per_minute": round((total_docs / wall_time) * 60, 2) if wall_time > 0 else 0,
        "worker_stats": worker_stats,
    }

    print(f"\n{'=' * 60}")
    print(f"{total_docs} docs, {total_pages} pages in {wall_time:.1f}s")
    print(f"{summary['pages_per_second']} pages/sec, {summary['docs_per_minute']} docs/min")
    print(f"Workers: {total_workers} ({workers_per_gpu}/GPU x {num_gpus} GPUs)")

    return summary
