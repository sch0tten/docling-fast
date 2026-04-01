"""Benchmark runner — grid search over batch sizes on GPU."""

import argparse
import json
import logging
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

from .config import AppConfig, load_config
from .extract import _count_doc
from .metrics import (
    BatchMetrics,
    DocMetrics,
    get_gpu_utilization,
    get_peak_vram_mb,
    get_system_info,
    reset_vram_stats,
    timer,
)
from .pipeline import build_converter, configure_docling_settings, require_cuda

_log = logging.getLogger(__name__)


def _collect_pdfs(cfg: AppConfig) -> list[Path]:
    samples_dir = Path(cfg.samples.dir)
    if not samples_dir.is_absolute():
        samples_dir = Path(__file__).parent.parent / samples_dir
    return sorted(p for p in samples_dir.iterdir() if p.suffix.lower() == ".pdf")


def _run_config(
    cfg: AppConfig,
    pdf_paths: list[Path],
    ocr_batch: int,
    layout_batch: int,
    table_batch: int,
    warmup: int = 2,
) -> dict:
    """Run extraction with a specific batch config, return metrics dict."""
    label = f"ocr_batch={ocr_batch} layout_batch={layout_batch} table_batch={table_batch}"
    print(f"\n--- {label} ---")

    reset_vram_stats()

    converter = build_converter(
        cfg,
        ocr_batch_override=ocr_batch,
        layout_batch_override=layout_batch,
        table_batch_override=table_batch,
    )

    metrics = BatchMetrics()
    gpu_utils = []

    # Warmup
    if warmup > 0:
        print(f"  Warmup: {min(warmup, len(pdf_paths))} docs...", end=" ", flush=True)
        for wp in pdf_paths[:warmup]:
            try:
                converter.convert(str(wp))
            except Exception:
                pass
        print("done")

    run_paths = pdf_paths[:cfg.benchmark.max_docs]
    print(f"  Processing {len(run_paths)} docs...")

    with timer() as batch_timer:
        for i, pdf_path in enumerate(run_paths, 1):
            doc_metrics = DocMetrics(source=pdf_path.name)

            with timer() as doc_timer:
                try:
                    result = converter.convert(str(pdf_path))
                    doc_dict = result.document.export_to_dict()
                    doc_metrics.pages, doc_metrics.tables, doc_metrics.text_length = _count_doc(doc_dict)
                    doc_metrics.success = True
                except Exception as e:
                    doc_metrics.success = False
                    doc_metrics.error = str(e)

            doc_metrics.wall_time_s = round(doc_timer["elapsed"], 3)
            gpu_util = get_gpu_utilization()
            gpu_utils.append(gpu_util)

            status = "OK" if doc_metrics.success else "FAIL"
            print(f"    [{i}/{len(run_paths)}] {pdf_path.name}: "
                  f"{status} {doc_metrics.wall_time_s:.1f}s {doc_metrics.pages}p GPU:{gpu_util:.0f}%")

            metrics.total_docs += 1
            if doc_metrics.success:
                metrics.successful_docs += 1
                metrics.total_pages += doc_metrics.pages
            else:
                metrics.failed_docs += 1

            metrics.per_doc.append({
                "source": doc_metrics.source,
                "pages": doc_metrics.pages,
                "time_s": doc_metrics.wall_time_s,
                "success": doc_metrics.success,
                "gpu_util_pct": gpu_util,
            })

    metrics.total_time_s = round(batch_timer["elapsed"], 2)
    metrics.peak_vram_mb = get_peak_vram_mb()
    metrics.avg_gpu_util_pct = round(sum(gpu_utils) / len(gpu_utils), 1) if gpu_utils else 0.0
    metrics.finalize()

    print(f"  -> {metrics.pages_per_second} pages/s, {metrics.docs_per_minute} docs/min, "
          f"VRAM {metrics.peak_vram_mb}MB, GPU {metrics.avg_gpu_util_pct}%")

    return {
        "config": {
            "ocr_backend": cfg.pipeline.ocr_backend,
            "ocr_batch_size": ocr_batch,
            "layout_batch_size": layout_batch,
            "table_batch_size": table_batch,
        },
        "results": metrics.to_dict(),
    }


def run_benchmark(cfg: AppConfig) -> dict:
    """Run GPU benchmark — grid search over batch sizes."""
    configure_docling_settings(cfg)

    pdf_paths = _collect_pdfs(cfg)
    if not pdf_paths:
        print("ERROR: No PDFs found in samples directory.")
        return {}

    system_info = get_system_info()
    print(f"Benchmarking on {system_info.get('gpu', '?')} ({system_info.get('vram_gb', '?')} GB)")
    print(f"{len(pdf_paths)} PDFs available, max {cfg.benchmark.max_docs} per config")

    configs_tested = []

    for ocr_bs, layout_bs, table_bs in product(
        cfg.benchmark.test_ocr_batch_sizes,
        cfg.benchmark.test_layout_batch_sizes,
        cfg.benchmark.test_table_batch_sizes,
    ):
        try:
            result = _run_config(cfg, pdf_paths, ocr_bs, layout_bs, table_bs, cfg.benchmark.warmup_docs)
            configs_tested.append(result)
        except Exception as e:
            print(f"  FAILED ocr={ocr_bs} layout={layout_bs} table={table_bs}: {e}")
            configs_tested.append({
                "config": {"ocr_batch_size": ocr_bs, "layout_batch_size": layout_bs, "table_batch_size": table_bs},
                "error": str(e),
            })

    successful = [c for c in configs_tested if "results" in c]
    best = max(successful, key=lambda c: c["results"]["pages_per_second"]) if successful else None

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    gpu_slug = system_info.get("gpu", "unknown").replace(" ", "_").lower()

    return {
        "run_id": f"{timestamp}_{gpu_slug}",
        "timestamp": timestamp,
        "system": system_info,
        "total_pdfs": len(pdf_paths),
        "configs_tested": configs_tested,
        "best_config": best["config"] if best else None,
        "best_pages_per_second": best["results"]["pages_per_second"] if best else None,
    }


def main():
    parser = argparse.ArgumentParser(prog="docling-bench", description="Benchmark Docling GPU extraction")
    parser.add_argument("-c", "--config", default=None, help="Config TOML file")
    parser.add_argument("-p", "--profile", default=None, help="Hardware profile name or TOML")
    parser.add_argument("--max-docs", type=int, default=None, help="Override max docs per config")
    parser.add_argument("--quick", action="store_true", help="Quick run: fewer configs, 5 docs")
    parser.add_argument("-o", "--output", default=None, help="Output JSON file path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    require_cuda()

    overrides = {}
    if args.max_docs:
        overrides["benchmark"] = {"max_docs": args.max_docs}
    if args.quick:
        overrides.setdefault("benchmark", {}).update({
            "max_docs": 5,
            "test_ocr_batch_sizes": [4, 32],
            "test_layout_batch_sizes": [4, 32],
            "test_table_batch_sizes": [4],
            "warmup_docs": 1,
        })

    cfg = load_config(config_path=args.config, hardware_profile=args.profile, overrides=overrides)
    report = run_benchmark(cfg)

    if not report:
        return

    output_dir = Path(cfg.output.dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.output) if args.output else output_dir / f"benchmark_{report['run_id']}.json"

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results: {out_path}")
    if report.get("best_config"):
        bc = report["best_config"]
        print(f"Best: ocr={bc['ocr_batch_size']} layout={bc['layout_batch_size']} "
              f"table={bc['table_batch_size']} -> {report['best_pages_per_second']} pages/sec")


if __name__ == "__main__":
    main()
