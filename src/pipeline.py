"""Build Docling DocumentConverter with GPU configuration."""

import os

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from .config import AppConfig


def require_cuda():
    """Assert CUDA is available. Call before building any converter."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. docling-fast requires a GPU. "
            "Check nvidia-smi and ensure PyTorch was installed with CUDA support."
        )


def configure_docling_settings(cfg: AppConfig) -> None:
    """Set Docling perf env vars. Must be called before importing docling pipelines."""
    os.environ["DOCLING_PERF_DOC_BATCH_SIZE"] = str(cfg.batch.doc_batch_size)
    os.environ["DOCLING_PERF_PAGE_BATCH_SIZE"] = str(cfg.batch.page_batch_size)


def build_converter(
    cfg: AppConfig,
    ocr_batch_override: int | None = None,
    layout_batch_override: int | None = None,
    table_batch_override: int | None = None,
) -> DocumentConverter:
    """Build a GPU-accelerated DocumentConverter.

    Args:
        cfg: Application config (device, batch sizes, OCR backend, etc.)
        ocr_batch_override: Override OCR batch size (for benchmarking)
        layout_batch_override: Override layout batch size (for benchmarking)
        table_batch_override: Override table batch size (for benchmarking)

    Raises:
        RuntimeError: If CUDA is not available.
    """
    require_cuda()

    ocr_batch = ocr_batch_override or cfg.pipeline.ocr_batch_size
    layout_batch = layout_batch_override or cfg.pipeline.layout_batch_size
    table_batch = table_batch_override or cfg.pipeline.table_batch_size

    accelerator = AcceleratorOptions(
        device=cfg.pipeline.device,
        num_threads=cfg.pipeline.num_threads,
    )

    ocr_options = RapidOcrOptions(
        backend=cfg.pipeline.ocr_backend,
        text_score=cfg.pipeline.ocr_text_score,
        force_full_page_ocr=cfg.pipeline.force_full_page_ocr,
    )

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=ocr_options,
        ocr_batch_size=ocr_batch,
        layout_batch_size=layout_batch,
        table_batch_size=table_batch,
        do_table_structure=cfg.pipeline.do_table_structure,
        accelerator_options=accelerator,
    )

    if cfg.pipeline.document_timeout > 0:
        pipeline_options.document_timeout = cfg.pipeline.document_timeout

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )
