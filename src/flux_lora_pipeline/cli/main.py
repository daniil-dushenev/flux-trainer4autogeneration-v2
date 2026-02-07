from __future__ import annotations

from pathlib import Path
import typer

from flux_lora_pipeline.core.io import load_config
from flux_lora_pipeline.data.dataset import validate_images
from flux_lora_pipeline.train.flux_backend import train_lora
from flux_lora_pipeline.gen.flux_sampler import generate_synthetic

app = typer.Typer(add_completion=False)


def _run_with_error_handling(fn) -> None:
    try:
        fn()
    except Exception as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command()
def train(config: Path = typer.Option(..., "--config", "-c")) -> None:
    def _run():
        cfg = load_config(config)
        validate_images(cfg.data.images_dir)
        train_lora(cfg)

    _run_with_error_handling(_run)


@app.command()
def generate(config: Path = typer.Option(..., "--config", "-c")) -> None:
    def _run():
        cfg = load_config(config)
        validate_images(cfg.data.images_dir)
        generate_synthetic(cfg)

    _run_with_error_handling(_run)


@app.command()
def all(config: Path = typer.Option(..., "--config", "-c")) -> None:
    def _run():
        cfg = load_config(config)
        validate_images(cfg.data.images_dir)
        train_lora(cfg)
        generate_synthetic(cfg)

    _run_with_error_handling(_run)


if __name__ == "__main__":
    app()
