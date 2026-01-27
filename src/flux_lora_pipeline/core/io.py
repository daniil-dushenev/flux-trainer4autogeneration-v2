from __future__ import annotations

from pathlib import Path
import yaml

from .config import PipelineConfig


def load_config(path: Path) -> PipelineConfig:
    data = yaml.safe_load(path.read_text())
    return PipelineConfig.model_validate(data)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
