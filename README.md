# Flux LoRA Pipeline (PoC)

Минимальный PoC пайплайна обучения LoRA (Flux) и генерации синтетики того же класса.

Сейчас backend по умолчанию `mock`: тренировка создает плейсхолдер LoRA, генерация делает простые аугментации, чтобы получить «другие» изображения того же класса. Это нужно, чтобы проверить, что пайплайн работает end-to-end.

Для реальной генерации добавлен backend `diffusers_flux2` (локально) и `fal_flux2` (через fal.ai).

## Структура
- `configs/base.yaml` — пример конфига
- `src/flux_lora_pipeline/train/flux_backend.py` — тренировка (mock или внешний скрипт)
- `src/flux_lora_pipeline/gen/flux_sampler.py` — генерация (mock или diffusers Flux2)
- `src/flux_lora_pipeline/cli/main.py` — CLI

## Запуск
1. Положи изображения одного класса в `/root/cats` или поменяй `data.images_dir` в конфиге.
2. Отредактируй `configs/base.yaml`.
3. Запусти:
   - `fluxpipe train -c configs/base.yaml`
   - `fluxpipe generate -c configs/base.yaml`
   - или `fluxpipe all -c configs/base.yaml`

## Backend'ы
### mock
Работает без GPU и весов. Генерация — аугментации. Тренировка — плейсхолдер.

### diffusers_flux2
Локальный backend через diffusers.
Тренировка LoRA: `train_dreambooth_lora_flux2_klein.py` из diffusers (используется в pipeline).
Генерация: `Flux2KleinPipeline`.

Требования:
- доступ в интернет для скачивания модели с Hugging Face
- рабочий GPU (CUDA) для приемлемой скорости
- `diffusers` из main (>=0.37.0.dev0), `transformers>=5.0.0`, `peft>=0.17.0`

### fal_flux2
Реальная тренировка LoRA через `fal-ai/flux-2-trainer-v2` и генерация через `fal-ai/flux-2/klein/4b/base/lora`.
Нужен `FAL_KEY` в окружении и сеть. LoRA возвращается как URL.
Важно: тренер `flux-2-trainer-v2` обучает LoRA для FLUX.2 [dev]. Совместимость с [klein] не гарантирована — при проблемах укажи совместимый inference endpoint.
