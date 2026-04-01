"""TOML configuration with Pydantic validation and cascading overrides."""

import sys
from pathlib import Path
from typing import Optional

import tomli
from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    device: str = "cuda:0"
    num_threads: int = 4
    ocr_backend: str = "torch"
    ocr_batch_size: int = 4
    layout_batch_size: int = 4
    table_batch_size: int = 4
    do_table_structure: bool = True
    force_full_page_ocr: bool = False
    ocr_text_score: float = 0.5
    document_timeout: float = 120.0


class BatchConfig(BaseModel):
    doc_batch_size: int = 1
    page_batch_size: int = 4


class WorkersConfig(BaseModel):
    num_gpus: int = 0           # 0 = auto-detect all available GPUs
    workers_per_gpu: int = 1    # worker processes per GPU


class BenchmarkConfig(BaseModel):
    test_ocr_batch_sizes: list[int] = Field(default_factory=lambda: [4, 8, 16, 32, 48, 64])
    test_layout_batch_sizes: list[int] = Field(default_factory=lambda: [4, 8, 16, 32])
    test_table_batch_sizes: list[int] = Field(default_factory=lambda: [4, 16])
    warmup_docs: int = 2
    max_docs: int = 50


class OutputConfig(BaseModel):
    format: str = "json"
    dir: str = "results"


class SamplesConfig(BaseModel):
    dir: str = "samples"


class AppConfig(BaseModel):
    pipeline: PipelineConfig = PipelineConfig()
    batch: BatchConfig = BatchConfig()
    workers: WorkersConfig = WorkersConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
    output: OutputConfig = OutputConfig()
    samples: SamplesConfig = SamplesConfig()


def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_path: Optional[str | Path] = None,
    hardware_profile: Optional[str | Path] = None,
    overrides: Optional[dict] = None,
) -> AppConfig:
    """Load config with cascading: default.toml -> hardware profile -> overrides.

    Args:
        config_path: Path to base config TOML (defaults to config/default.toml)
        hardware_profile: Path to hardware-specific TOML overlay, or a name
                          that resolves to config/{name}.toml
        overrides: Dict of runtime overrides (e.g. from CLI args)
    """
    project_root = Path(__file__).parent.parent

    default_path = config_path or project_root / "config" / "default.toml"
    default_path = Path(default_path)
    data = {}
    if default_path.exists():
        with open(default_path, "rb") as f:
            data = tomli.load(f)

    if hardware_profile:
        hw_path = Path(hardware_profile)
        if not hw_path.exists():
            hw_path = project_root / "config" / f"{hardware_profile}.toml"
        if hw_path.exists():
            with open(hw_path, "rb") as f:
                data = _deep_merge(data, tomli.load(f))

    if overrides:
        data = _deep_merge(data, overrides)

    return AppConfig(**data)


def main():
    import json

    profile = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = load_config(hardware_profile=profile)
    print(json.dumps(cfg.model_dump(), indent=2))


if __name__ == "__main__":
    main()
