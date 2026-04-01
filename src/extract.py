"""PDF extraction to JSON using Docling with GPU acceleration."""

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config
from .metrics import DocMetrics, timer
from .pipeline import build_converter, configure_docling_settings, require_cuda

_log = logging.getLogger(__name__)


def _count_doc(doc_dict: dict) -> tuple[int, int, int]:
    """Extract page count, table count, and text length from a docling dict."""
    pages = doc_dict.get("pages", {})
    page_count = len(pages) if isinstance(pages, (dict, list)) else 0

    tables = doc_dict.get("tables", [])
    table_count = len(tables) if isinstance(tables, (dict, list)) else 0

    texts = doc_dict.get("texts", [])
    text_length = sum(len(t.get("text", "")) for t in texts if isinstance(t, dict)) if isinstance(texts, list) else 0

    return page_count, table_count, text_length


def extract_one(pdf_path: Path, converter) -> dict:
    """Extract a single PDF to a JSON-serializable dict.

    Returns dict with keys: source, pages, extraction_time_s, text_length,
    tables, success, and (on success) document.
    """
    metrics = DocMetrics(source=pdf_path.name)

    with timer() as t:
        try:
            result = converter.convert(str(pdf_path))
            doc_dict = result.document.export_to_dict()
            metrics.pages, metrics.tables, metrics.text_length = _count_doc(doc_dict)
        except Exception as e:
            _log.error(f"Failed to extract {pdf_path.name}: {e}")
            metrics.success = False
            metrics.error = str(e)
            doc_dict = None

    metrics.wall_time_s = round(t["elapsed"], 3)

    output = {
        "source": metrics.source,
        "pages": metrics.pages,
        "extraction_time_s": metrics.wall_time_s,
        "text_length": metrics.text_length,
        "tables": metrics.tables,
        "success": metrics.success,
    }
    if metrics.error:
        output["error"] = metrics.error
    if doc_dict is not None:
        output["document"] = doc_dict

    return output


def extract_batch(pdf_paths: list[Path], converter) -> list[dict]:
    """Extract multiple PDFs via convert_all, returning a list of result dicts."""
    results = []
    sources = [str(p) for p in pdf_paths]

    try:
        for conv_result in converter.convert_all(sources):
            source_name = Path(conv_result.input.file.name).name if conv_result.input.file else "unknown"

            try:
                doc_dict = conv_result.document.export_to_dict()
                pages, tables, text_length = _count_doc(doc_dict)
                results.append({
                    "source": source_name,
                    "pages": pages,
                    "text_length": text_length,
                    "tables": tables,
                    "success": True,
                    "document": doc_dict,
                })
            except Exception as e:
                _log.error(f"Failed to process {source_name}: {e}")
                results.append({"source": source_name, "success": False, "error": str(e)})

    except Exception as e:
        _log.error(f"Batch conversion failed: {e}")
        for path in pdf_paths:
            if not any(r["source"] == path.name for r in results):
                results.append({"source": path.name, "success": False, "error": str(e)})

    return results


def main():
    parser = argparse.ArgumentParser(prog="docling-fast", description="GPU-accelerated PDF extraction to JSON")
    parser.add_argument("-i", "--input", required=True, help="PDF file or directory of PDFs")
    parser.add_argument("-o", "--output", default=None, help="Output directory (default: results/)")
    parser.add_argument("-c", "--config", default=None, help="Config TOML file")
    parser.add_argument("-p", "--profile", default=None, help="Hardware profile name or TOML path")
    parser.add_argument("--single-file", action="store_true", help="Write all results to one JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    require_cuda()

    cfg = load_config(config_path=args.config, hardware_profile=args.profile)
    configure_docling_settings(cfg)

    output_dir = Path(args.output) if args.output else Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect PDFs
    input_path = Path(args.input)
    if input_path.is_file():
        pdf_paths = [input_path]
    elif input_path.is_dir():
        pdf_paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() == ".pdf")
    else:
        print(f"ERROR: {input_path} is not a file or directory", file=sys.stderr)
        sys.exit(1)

    if not pdf_paths:
        print(f"ERROR: No PDFs found in {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(pdf_paths)} PDFs (device={cfg.pipeline.device}, "
          f"ocr_batch={cfg.pipeline.ocr_batch_size})")

    converter = build_converter(cfg)

    all_results = []
    for i, pdf_path in enumerate(pdf_paths, 1):
        print(f"  [{i}/{len(pdf_paths)}] {pdf_path.name}...", end=" ", flush=True)
        result = extract_one(pdf_path, converter)
        all_results.append(result)

        status = "OK" if result["success"] else f"FAIL: {result.get('error', '?')}"
        print(f"{status} ({result.get('extraction_time_s', 0):.1f}s, {result.get('pages', 0)}p)")

        if not args.single_file:
            with open(output_dir / f"{pdf_path.stem}.json", "w") as f:
                json.dump(result, f, indent=2)

    if args.single_file:
        out_file = output_dir / "extraction_results.json"
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults: {out_file}")
    else:
        print(f"\nResults: {output_dir}/ ({len(all_results)} files)")

    ok = sum(1 for r in all_results if r["success"])
    total_pages = sum(r.get("pages", 0) for r in all_results)
    total_time = sum(r.get("extraction_time_s", 0) for r in all_results)
    print(f"{ok}/{len(all_results)} succeeded, {total_pages} pages, {total_time:.1f}s")
    if total_time > 0:
        print(f"{total_pages / total_time:.1f} pages/sec, {(ok / total_time) * 60:.1f} docs/min")


if __name__ == "__main__":
    main()
