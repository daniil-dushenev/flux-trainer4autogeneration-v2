from __future__ import annotations

from pathlib import Path
import typer

from flux_lora_pipeline.core.io import load_config
from flux_lora_pipeline.data.dataset import validate_images
from flux_lora_pipeline.train.flux_backend import train_lora
from flux_lora_pipeline.gen.flux_sampler import generate_synthetic

app = typer.Typer(add_completion=False)


@app.command()
def train(config: Path = typer.Option(..., "--config", "-c")) -> None:
    cfg = load_config(config)
    validate_images(cfg.data.images_dir)
    train_lora(cfg)


@app.command()
def generate(config: Path = typer.Option(..., "--config", "-c")) -> None:
    cfg = load_config(config)
    validate_images(cfg.data.images_dir)
    generate_synthetic(cfg)


@app.command()
def all(config: Path = typer.Option(..., "--config", "-c")) -> None:
    cfg = load_config(config)
    validate_images(cfg.data.images_dir)
    train_lora(cfg)
    generate_synthetic(cfg)


if __name__ == "__main__":
    app()
