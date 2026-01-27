from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    images_dir: Path = Field(..., description="Path to class images folder")
    captions_dir: Path | None = Field(None, description="Optional captions folder")
    resolution: int = 768
    repeats: int = 1


class TrainConfig(BaseModel):
    output_dir: Path = Path("outputs")
    lora_rank: int = 16
    learning_rate: float = 1e-4
    steps: int = 1000
    batch_size: int = 1
    seed: int = 42
    command: str | None = None
    command_workdir: Path | None = None


class GenerateConfig(BaseModel):
    out_dir: Path = Path("synthetic")
    num_images: int = 100
    guidance_scale: float = 3.5
    steps: int = 30
    seed: int = 123
    height: int = 768
    width: int = 768
    lora_path: Path | None = None
    lora_scale: float = 1.0


class FluxConfig(BaseModel):
    backend: str = "mock"  # mock | diffusers_flux2 | fal_flux2 | external
    model_path: str = ""
    transformer_path: str | None = None
    revision: str | None = None
    dtype: str = "bf16"
    device: str = "cuda"
    enable_cpu_offload: bool = False


class DiffusersTrainConfig(BaseModel):
    script_path: Path = Path(
        ".third_party/diffusers/examples/dreambooth/train_dreambooth_lora_flux2_klein.py"
    )
    pretrained_model_name_or_path: str = "black-forest-labs/FLUX.2-klein-base-4B"
    resolution: int = 512
    train_batch_size: int = 1
    max_train_steps: int = 50
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    mixed_precision: str = "bf16"
    repeats: int = 10
    seed: int = 42
    output_dir: Path = Path("outputs")


class FalTrainConfig(BaseModel):
    app: str = "fal-ai/flux-2-trainer-v2"
    steps: int = 1000
    learning_rate: float = 5e-5
    default_caption: str = "a photo of {class_name}"
    output_lora_format: str = "fal"  # fal | comfy
    data_zip: Path | None = None


class FalGenerateConfig(BaseModel):
    app: str = "fal-ai/flux-2/klein/4b/base/lora"
    lora_path: str | None = None
    num_images: int = 4
    guidance_scale: float = 3.5
    steps: int = 30
    height: int = 768
    width: int = 768


class PipelineConfig(BaseModel):
    project_name: str = "class-pipeline"
    class_name: str = "class"
    class_prompt: str = "a photo of {class_name}"
    data: DataConfig
    train: TrainConfig
    generate: GenerateConfig
    flux: FluxConfig
    diffusers_train: DiffusersTrainConfig = DiffusersTrainConfig()
    fal_train: FalTrainConfig = FalTrainConfig()
    fal_generate: FalGenerateConfig = FalGenerateConfig()
