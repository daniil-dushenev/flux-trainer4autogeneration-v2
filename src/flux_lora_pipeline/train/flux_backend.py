from __future__ import annotations

from pathlib import Path
import json
import os
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

    if cfg.flux.backend == "diffusers_flux2":
        return _train_diffusers_flux2_klein(cfg)

    if cfg.flux.backend == "fal_flux2":
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


def _train_diffusers_flux2_klein(cfg: PipelineConfig) -> Path:
    script_path = cfg.diffusers_train.script_path
    if not script_path.exists():
        raise FileNotFoundError(f"Diffusers training script not found: {script_path}")

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
        "--repeats",
        str(cfg.diffusers_train.repeats),
        "--seed",
        str(cfg.diffusers_train.seed),
        "--skip_final_inference",
    ]
    if cfg.diffusers_train.gradient_checkpointing:
        args.append("--gradient_checkpointing")
    if cfg.diffusers_train.cache_latents:
        args.append("--cache_latents")

    subprocess.run(args, check=True, env=env)

    # Diffusers saves LoRA weights inside output_dir
    return cfg.train.output_dir / "pytorch_lora_weights.safetensors"


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

    lora_file = result.get("diffusers_lora_file") or result.get("lora_file")
    if isinstance(lora_file, dict):
        lora_url = lora_file.get("url")
    else:
        lora_url = lora_file
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
