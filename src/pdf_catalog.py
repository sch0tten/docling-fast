"""Scan PDF directories, classify OCR-heavy vs programmatic text, build sample set."""

import argparse
import json
from pathlib import Path


def classify_pdf(pdf_path: Path) -> dict:
    """Classify a PDF as OCR-heavy or text-rich.

    Uses PyMuPDF to check embedded text per page.
    OCR-heavy = avg < 100 chars of embedded text per page.
    """
    import fitz

    try:
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        if page_count == 0:
            doc.close()
            return {"name": pdf_path.name, "path": str(pdf_path.resolve()), "pages": 0, "type": "empty"}

        total_chars = sum(len(page.get_text().strip()) for page in doc)
        avg_chars = total_chars / page_count
        file_size_kb = pdf_path.stat().st_size // 1024
        doc.close()

        return {
            "name": pdf_path.name,
            "path": str(pdf_path.resolve()),
            "pages": page_count,
            "type": "ocr_heavy" if avg_chars < 100 else "text_rich",
            "avg_chars_per_page": round(avg_chars, 1),
            "file_size_kb": file_size_kb,
        }
    except Exception as e:
        return {"name": pdf_path.name, "path": str(pdf_path.resolve()), "pages": 0, "type": "error", "error": str(e)}


def scan_directories(dirs: list[Path]) -> list[dict]:
    """Scan directories for PDFs and classify them."""
    all_pdfs = []
    for d in dirs:
        if not d.exists():
            continue
        for pdf in sorted(d.rglob("*.pdf")):
            if pdf.is_file():
                all_pdfs.append(pdf)

    print(f"Found {len(all_pdfs)} PDFs across {len(dirs)} directories")

    catalog = []
    for i, pdf in enumerate(all_pdfs, 1):
        if i % 50 == 0:
            print(f"  Classified {i}/{len(all_pdfs)}...")
        catalog.append(classify_pdf(pdf))

    return catalog


def build_sample_set(catalog: list[dict], samples_dir: Path, max_samples: int = 50) -> list[dict]:
    """Select OCR-heavy PDFs and create symlinks in samples_dir."""
    ocr_heavy = sorted(
        [e for e in catalog if e["type"] == "ocr_heavy" and e["pages"] > 0],
        key=lambda e: e["pages"], reverse=True,
    )

    # Fill with text_rich if not enough OCR-heavy
    if len(ocr_heavy) < max_samples:
        text_rich = sorted(
            [e for e in catalog if e["type"] == "text_rich" and e["pages"] > 0],
            key=lambda e: e["pages"], reverse=True,
        )
        ocr_heavy.extend(text_rich)

    selected = ocr_heavy[:max_samples]
    ocr_count = sum(1 for s in selected if s["type"] == "ocr_heavy")
    total_pages = sum(s["pages"] for s in selected)
    print(f"Selected {len(selected)} PDFs ({ocr_count} OCR-heavy, {total_pages} pages)")

    samples_dir.mkdir(parents=True, exist_ok=True)

    # Clear old symlinks
    for existing in samples_dir.glob("*.pdf"):
        if existing.is_symlink():
            existing.unlink()

    for entry in selected:
        src = Path(entry["path"])
        dst = samples_dir / entry["name"]
        if dst.exists():
            dst = samples_dir / f"{src.parent.name}_{entry['name']}"
        try:
            dst.symlink_to(src)
        except OSError as e:
            print(f"  WARNING: Could not symlink {entry['name']}: {e}")

    return selected


def main():
    parser = argparse.ArgumentParser(prog="pdf-catalog", description="Classify PDFs and build OCR-heavy sample set")
    parser.add_argument("scan_dirs", nargs="+", help="Directories to scan for PDFs")
    parser.add_argument("--samples-dir", default=None, help="Output dir for symlinks (default: samples/)")
    parser.add_argument("--max", type=int, default=50, help="Max samples to select")
    parser.add_argument("--catalog-only", action="store_true", help="Only write catalog, skip symlinks")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    scan_dirs = [Path(d) for d in args.scan_dirs]
    samples_dir = Path(args.samples_dir) if args.samples_dir else project_root / "samples"

    catalog = scan_directories(scan_dirs)

    catalog_path = samples_dir / "catalog.json"
    samples_dir.mkdir(parents=True, exist_ok=True)
    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"Catalog: {catalog_path} ({len(catalog)} entries)")

    by_type = {}
    for entry in catalog:
        by_type.setdefault(entry["type"], []).append(entry)
    for t, entries in sorted(by_type.items()):
        print(f"  {t}: {len(entries)} PDFs, {sum(e['pages'] for e in entries)} pages")

    if not args.catalog_only:
        selected = build_sample_set(catalog, samples_dir, args.max)
        with open(samples_dir / "selected.json", "w") as f:
            json.dump(selected, f, indent=2)


if __name__ == "__main__":
    main()
