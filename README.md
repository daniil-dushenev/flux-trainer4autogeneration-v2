# Flux LoRA Pipeline (FLUX.2 [dev], local A100)

Пайплайн обучения LoRA и генерации синтетики для одного класса на базе `FLUX.2 [dev]`.

Основной backend по умолчанию: `diffusers_flux2_dev` (полностью локально, без fal.ai).
`mock` остается для smoke-теста без загрузки модели.

## Структура
- `configs/base.yaml` — пример конфига
- `src/flux_lora_pipeline/train/flux_backend.py` — тренировка (mock, fal или diffusers)
- `src/flux_lora_pipeline/gen/flux_sampler.py` — генерация (mock, fal или diffusers)
- `src/flux_lora_pipeline/cli/main.py` — CLI

## Запуск
1. Положи изображения одного класса в `/root/MT_Crack/Imgs` или поменяй `data.images_dir` в конфиге.
2. Отредактируй `configs/base.yaml`.
3. Запусти:
   - `fluxpipe train -c configs/base.yaml`
   - `fluxpipe generate -c configs/base.yaml`
   - или `fluxpipe all -c configs/base.yaml`

## Backend'ы
### mock
Работает без GPU и без API. Генерация делает аугментации, тренировка создает плейсхолдер LoRA.

### diffusers_flux2_dev
Полностью локальный backend через diffusers:
- train: `train_dreambooth_lora_flux2.py`
- generate: `Flux2Pipeline`

### fal_flux2_dev (опционально)
Удаленный backend через fal.ai. Не нужен для локального режима.

## Быстрый старт (FLUX.2 [dev])
1. Укажи путь к датасету в `configs/base.yaml` (`data.images_dir`).
2. Прими gated-модель и залогинься в HF:
   - `hf auth login`
3. Подними воспроизводимое окружение:
   - `bash scripts/setup_env.sh`
4. Активируй окружение:
   - `source .venv/bin/activate`
5. Запуск:
   - `fluxpipe train -c configs/base.yaml`
   - `fluxpipe generate -c configs/base.yaml`
   - или `fluxpipe all -c configs/base.yaml`

## Зависимости
- Зафиксированные версии: `requirements/lock.txt`
- Автоматическая установка под проект: `scripts/setup_env.sh`
- Скрипт ставит `diffusers` из commit `10dc589a942a38a53d598275b0b421b69a6e03cc` и затем ставит проект в editable-режиме.
- Если хочешь установить вручную:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -U pip setuptools wheel`
  - `pip install -r requirements/lock.txt`
  - `git clone https://github.com/huggingface/diffusers.git .third_party/diffusers`
  - `git -C .third_party/diffusers checkout 10dc589a942a38a53d598275b0b421b69a6e03cc`
  - `pip install --no-deps -e .third_party/diffusers`
  - `pip install --no-deps -e .`

## OOM заметки (A100)
- `FLUX.2-dev` очень тяжелый даже для 80GB VRAM.
- В `configs/base.yaml` уже включены memory-safe параметры:
  - `diffusers_train.offload: true`
  - `diffusers_train.use_8bit_adam: true`
  - `diffusers_train.gradient_checkpointing: true`
  - `diffusers_train.cache_latents: true`
  - `diffusers_train.resolution: 384`
  - `train.lora_rank: 8`
  - `diffusers_train.bnb_quantization_config_path: "configs/bnb_4bit_flux2.json"` (NF4 4-bit для transformer)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` выставляется автоматически в train backend.
- Для A100 не включай `--do_fp8_training`: в diffusers это рассчитано на GPU с compute capability >= 8.9, у A100 обычно 8.0.

## Профиль для малого датасета (~50 изображений)
- `train.lora_rank=8`, `train.lora_alpha=8`, `train.lora_dropout=0.08`
- `diffusers_train.learning_rate=7e-5`, `diffusers_train.max_train_steps=600`
- `diffusers_train.gradient_accumulation_steps=2`
- `diffusers_train.lr_scheduler=constant_with_warmup`, `diffusers_train.lr_warmup_steps=60`
- `diffusers_train.repeats=1` (для ~50 изображений это ~12 эпох)
- `generate.lora_scale=1.0`, `generate.guidance_scale=2.5`
