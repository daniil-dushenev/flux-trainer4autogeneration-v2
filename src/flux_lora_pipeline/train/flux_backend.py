from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import subprocess
import time
import shutil

from flux_lora_pipeline.core.io import ensure_dir
from flux_lora_pipeline.core.config import PipelineConfig
from flux_lora_pipeline.data.zipper import make_dataset_zip
from flux_lora_pipeline.data.dataset import iter_images


def train_lora(cfg: PipelineConfig) -> Path:
    if cfg.flux.backend == "mock":
        ensure_dir(cfg.train.output_dir)
        out_path = cfg.train.output_dir / f"{cfg.project_name}_lora.safetensors"

        metadata = {
            "project": cfg.project_name,
            "class_name": cfg.class_name,
            "class_prompt": cfg.class_prompt.format(class_name=cfg.class_name),
            "steps": cfg.train.steps,
            "lora_rank": cfg.train.lora_rank,
            "created_at": int(time.time()),
            "backend": cfg.flux.backend,
            "note": "PoC placeholder artifact. Replace with real training output.",
        }
        (cfg.train.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        out_path.write_bytes(b"PSEUDO_LORA")
        return out_path

    if cfg.flux.backend in {"diffusers_flux2", "diffusers_flux2_dev"}:
        return _train_diffusers_flux2_dev(cfg)

    if cfg.flux.backend in {"fal_flux2", "fal_flux2_dev"}:
        return _train_fal_flux2(cfg)

    if cfg.flux.backend == "external":
        if not cfg.train.command:
            raise ValueError("train.command must be set for backend=external")
        workdir = cfg.train.command_workdir or Path(".")
        ensure_dir(cfg.train.output_dir)
        subprocess.run(cfg.train.command, shell=True, check=True, cwd=workdir)
        return cfg.train.output_dir / f"{cfg.project_name}_lora.safetensors"

    raise NotImplementedError(
        "Flux2 LoRA training backend not implemented. "
        "Use backend=external to call your training script."
    )


def _train_diffusers_flux2_dev(cfg: PipelineConfig) -> Path:
    script_path = cfg.diffusers_train.script_path
    if not script_path.exists():
        raise FileNotFoundError(
            f"Diffusers training script not found: {script_path}. "
            "Clone diffusers into .third_party/diffusers and install it in editable mode."
        )

    instance_prompt = cfg.class_prompt.format(class_name=cfg.class_name)
    ensure_dir(cfg.train.output_dir)
    instance_dir = cfg.train.output_dir / "_instance_images"
    if instance_dir.exists():
        shutil.rmtree(instance_dir)
    instance_dir.mkdir(parents=True, exist_ok=True)
    for p in iter_images(cfg.data.images_dir):
        shutil.copy2(p, instance_dir / p.name)

    env = os.environ.copy()
    env.setdefault("HF_HOME", "/root/.cache/huggingface")
    env.setdefault("TRANSFORMERS_CACHE", "/root/.cache/huggingface/hub")
    env.setdefault("ACCELERATE_CONFIG_FILE", "/root/.cache/accelerate/default_config.yaml")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _ensure_accelerate_config(Path(env["ACCELERATE_CONFIG_FILE"]), cfg.diffusers_train.mixed_precision)

    args = [
        "accelerate",
        "launch",
        "--config_file",
        env["ACCELERATE_CONFIG_FILE"],
        str(script_path),
        "--pretrained_model_name_or_path",
        cfg.diffusers_train.pretrained_model_name_or_path,
        "--instance_data_dir",
        str(instance_dir),
        "--output_dir",
        str(cfg.train.output_dir),
        "--instance_prompt",
        instance_prompt,
        "--resolution",
        str(cfg.diffusers_train.resolution),
        "--train_batch_size",
        str(cfg.diffusers_train.train_batch_size),
        "--max_train_steps",
        str(cfg.diffusers_train.max_train_steps),
        "--learning_rate",
        str(cfg.diffusers_train.learning_rate),
        "--gradient_accumulation_steps",
        str(cfg.diffusers_train.gradient_accumulation_steps),
        "--mixed_precision",
        cfg.diffusers_train.mixed_precision,
        "--guidance_scale",
        str(cfg.diffusers_train.guidance_scale),
        "--lr_scheduler",
        cfg.diffusers_train.lr_scheduler,
        "--lr_warmup_steps",
        str(cfg.diffusers_train.lr_warmup_steps),
        "--optimizer",
        cfg.diffusers_train.optimizer,
        "--repeats",
        str(cfg.diffusers_train.repeats),
        "--rank",
        str(cfg.train.lora_rank),
        "--lora_alpha",
        str(cfg.train.lora_alpha if cfg.train.lora_alpha is not None else cfg.train.lora_rank),
        "--lora_dropout",
        str(cfg.train.lora_dropout),
        "--seed",
        str(cfg.diffusers_train.seed),
        "--skip_final_inference",
    ]
    if cfg.diffusers_train.gradient_checkpointing:
        args.append("--gradient_checkpointing")
    if cfg.diffusers_train.cache_latents:
        args.append("--cache_latents")
    if cfg.diffusers_train.offload:
        args.append("--offload")
    if cfg.diffusers_train.remote_text_encoder:
        args.append("--remote_text_encoder")
    if cfg.diffusers_train.use_8bit_adam:
        args.append("--use_8bit_adam")
    if cfg.diffusers_train.bnb_quantization_config_path:
        args.extend(
            [
                "--bnb_quantization_config_path",
                str(cfg.diffusers_train.bnb_quantization_config_path),
            ]
        )

    subprocess.run(args, check=True, env=env)

    # Diffusers saves LoRA weights inside output_dir
    return cfg.train.output_dir / "pytorch_lora_weights.safetensors"


def _ensure_accelerate_config(config_path: Path, mixed_precision: str) -> None:
    if config_path.exists():
        return
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from accelerate.utils import write_basic_config

        write_basic_config(mixed_precision=mixed_precision, save_location=str(config_path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create accelerate config at {config_path}. "
            "Run `accelerate config default` manually."
        ) from exc


def _train_fal_flux2(cfg: PipelineConfig) -> Path:
    if not os.environ.get("FAL_KEY"):
        raise RuntimeError("FAL_KEY is not set in the environment")

    import fal_client

    ensure_dir(cfg.train.output_dir)

    caption = cfg.fal_train.default_caption.format(class_name=cfg.class_name)
    data_zip = cfg.fal_train.data_zip or (cfg.train.output_dir / "dataset.zip")
    make_dataset_zip(
        images_dir=cfg.data.images_dir,
        captions_dir=cfg.data.captions_dir,
        out_path=data_zip,
        default_caption=caption,
    )

    data_url = fal_client.upload_file(str(data_zip))

    args = {
        "image_data_url": data_url,
        "steps": cfg.fal_train.steps,
        "learning_rate": cfg.fal_train.learning_rate,
        "default_caption": caption,
        "output_lora_format": cfg.fal_train.output_lora_format,
    }

    result = fal_client.subscribe(cfg.fal_train.app, arguments=args)

    lora_url = _extract_lora_url(result)
    if not lora_url:
        raise RuntimeError(f"fal trainer returned no LoRA url: {result}")

    metadata = {
        "project": cfg.project_name,
        "class_name": cfg.class_name,
        "class_prompt": caption,
        "backend": cfg.flux.backend,
        "fal_app": cfg.fal_train.app,
        "lora_url": lora_url,
        "created_at": int(time.time()),
    }
    (cfg.train.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Save a tiny placeholder file that references the remote LoRA
    out_path = cfg.train.output_dir / f"{cfg.project_name}_lora.url.txt"
    out_path.write_text(lora_url)
    return out_path


def _extract_lora_url(result: dict[str, Any]) -> str | None:
    keys = ("diffusers_lora_file", "lora_file", "lora", "adapter", "weights")
    for key in keys:
        value = result.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
        if isinstance(value, dict):
            url = value.get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                return url

    for value in result.values():
        if isinstance(value, dict):
            nested = _extract_lora_url(value)
            if nested:
                return nested
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    nested = _extract_lora_url(item)
                    if nested:
                        return nested
                if isinstance(item, str) and item.startswith(("http://", "https://")):
                    return item
    return None
