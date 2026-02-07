from __future__ import annotations

import json
import os
import random

from pathlib import Path

from PIL import Image, ImageEnhance

from flux_lora_pipeline.core.config import PipelineConfig
from flux_lora_pipeline.core.io import ensure_dir
from flux_lora_pipeline.data.dataset import iter_images


def _augment(im: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    im = im.copy()

    # Simple PoC augmentations to keep the same class
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.2:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)

    # Color and contrast tweaks
    color = ImageEnhance.Color(im)
    im = color.enhance(rng.uniform(0.85, 1.15))
    contrast = ImageEnhance.Contrast(im)
    im = contrast.enhance(rng.uniform(0.9, 1.1))

    return im


def generate_synthetic(cfg: PipelineConfig) -> None:
    if cfg.flux.backend == "mock":
        ensure_dir(cfg.generate.out_dir)
        images = list(iter_images(cfg.data.images_dir))
        if not images:
            raise ValueError("No images for synthetic generation")

        rng = random.Random(cfg.generate.seed)
        for i in range(cfg.generate.num_images):
            src = rng.choice(images)
            with Image.open(src) as im:
                im = im.convert("RGB")
                out = _augment(im, seed=rng.randint(0, 1_000_000))
                out_path = cfg.generate.out_dir / f"synthetic_{i:05d}.jpg"
                out.save(out_path, quality=92)
        return

    if cfg.flux.backend in {"diffusers_flux2", "diffusers_flux2_dev"}:
        _generate_flux2_diffusers(cfg)
        return
    if cfg.flux.backend in {"fal_flux2", "fal_flux2_dev"}:
        _generate_flux2_fal(cfg)
        return

    raise ValueError(f"Unknown backend: {cfg.flux.backend}")


def _dtype_from_cfg(dtype: str):
    import torch

    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def _generate_flux2_diffusers(cfg: PipelineConfig) -> None:
    import torch
    from diffusers import BitsAndBytesConfig, Flux2Pipeline

    ensure_dir(cfg.generate.out_dir)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    dtype = _dtype_from_cfg(cfg.flux.dtype)

    if cfg.flux.transformer_path:
        try:
            from diffusers import Flux2Transformer2DModel as TransformerCls
        except Exception:
            from diffusers import FluxTransformer2DModel as TransformerCls

        transformer = TransformerCls.from_pretrained(cfg.flux.transformer_path, torch_dtype=dtype)
        pipe = Flux2Pipeline.from_pretrained(cfg.flux.model_path, transformer=transformer, torch_dtype=dtype)
    elif cfg.flux.use_bnb_4bit_for_inference and cfg.diffusers_train.bnb_quantization_config_path:
        try:
            from diffusers import Flux2Transformer2DModel as TransformerCls
        except Exception:
            from diffusers import FluxTransformer2DModel as TransformerCls

        with open(cfg.diffusers_train.bnb_quantization_config_path, "r") as f:
            quant_cfg_data = json.load(f)
        if quant_cfg_data.get("load_in_4bit"):
            quant_cfg_data["bnb_4bit_compute_dtype"] = dtype
        quantization_config = BitsAndBytesConfig(**quant_cfg_data)
        transformer = TransformerCls.from_pretrained(
            cfg.flux.model_path,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )
        pipe = Flux2Pipeline.from_pretrained(cfg.flux.model_path, transformer=transformer, torch_dtype=dtype)
    else:
        pipe = Flux2Pipeline.from_pretrained(cfg.flux.model_path, torch_dtype=dtype)

    if cfg.flux.enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(cfg.flux.device)

    if cfg.generate.lora_path:
        lora_path = Path(cfg.generate.lora_path)
        if lora_path.is_dir():
            safetensors = sorted(lora_path.glob("*.safetensors"))
            if not safetensors:
                safetensors = sorted(lora_path.rglob("*.safetensors"))
            if safetensors:
                chosen = next((p for p in safetensors if p.name == "pytorch_lora_weights.safetensors"), safetensors[0])
                pipe.load_lora_weights(str(chosen.parent), weight_name=chosen.name)
            else:
                pipe.load_lora_weights(str(lora_path))
        else:
            pipe.load_lora_weights(str(lora_path))
        if cfg.generate.lora_scale != 1.0 and hasattr(pipe, "set_adapters"):
            try:
                pipe.set_adapters(["default"], adapter_weights=[cfg.generate.lora_scale])
            except Exception:
                pass

    prompt = cfg.class_prompt.format(class_name=cfg.class_name)
    rng = torch.Generator(device=cfg.flux.device).manual_seed(cfg.generate.seed)
    for i in range(cfg.generate.num_images):
        image = pipe(
            prompt=prompt,
            guidance_scale=cfg.generate.guidance_scale,
            num_inference_steps=cfg.generate.steps,
            height=cfg.generate.height,
            width=cfg.generate.width,
            generator=rng,
        ).images[0]
        out_path = cfg.generate.out_dir / f"synthetic_{i:05d}.png"
        image.save(out_path)


def _generate_flux2_fal(cfg: PipelineConfig) -> None:
    import os
    import urllib.request
    import fal_client

    if not os.environ.get("FAL_KEY"):
        raise RuntimeError("FAL_KEY is not set in the environment")

    ensure_dir(cfg.generate.out_dir)

    lora_path = cfg.fal_generate.lora_path
    if not lora_path and cfg.generate.lora_path:
        lora_path = str(cfg.generate.lora_path)
    if not lora_path:
        fallback = cfg.train.output_dir / f"{cfg.project_name}_lora.url.txt"
        if fallback.exists():
            lora_path = fallback.read_text().strip()

    prompt = cfg.class_prompt.format(class_name=cfg.class_name)
    num_images = cfg.fal_generate.num_images or cfg.generate.num_images
    args = {
        "prompt": prompt,
        "image_size": {"height": cfg.fal_generate.height, "width": cfg.fal_generate.width},
        "guidance_scale": cfg.fal_generate.guidance_scale,
        "num_inference_steps": cfg.fal_generate.steps,
        "num_images": num_images,
        "seed": cfg.generate.seed,
    }
    if lora_path:
        args["loras"] = [{"path": lora_path, "scale": cfg.generate.lora_scale}]

    result = fal_client.subscribe(cfg.fal_generate.app, arguments=args)
    images = result.get("images", [])
    if not images:
        raise RuntimeError(f"fal generation returned no images: {result}")

    for i, item in enumerate(images):
        url = item.get("url")
        if not url:
            continue
        out_path = cfg.generate.out_dir / f"synthetic_{i:05d}.png"
        urllib.request.urlretrieve(url, out_path)
