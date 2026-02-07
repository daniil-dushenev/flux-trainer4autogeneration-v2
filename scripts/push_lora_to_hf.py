#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token


def _resolve_lora_file(path: Path) -> Path:
    if path.is_file():
        return path

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Path must be a file or directory: {path}")

    preferred = path / "pytorch_lora_weights.safetensors"
    if preferred.exists():
        return preferred

    candidates = sorted(path.rglob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No .safetensors file found in: {path}")
    return candidates[0]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _build_readme(
    repo_id: str,
    base_model: str,
    trigger_text: str | None,
    lora_filename: str,
    sha256: str,
) -> str:
    trigger_line = trigger_text if trigger_text else "Not specified"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    return f"""---
license: other
base_model: {base_model}
tags:
  - lora
  - flux
  - flux-2
  - diffusers
  - text-to-image
library_name: diffusers
---

# {repo_id}

FLUX.2-dev LoRA adapter.

## Files

- `{lora_filename}` (SHA256: `{sha256}`)

## Usage (Diffusers)

```python
import torch
from diffusers import Flux2Pipeline

pipe = Flux2Pipeline.from_pretrained("{base_model}", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("{repo_id}", weight_name="{lora_filename}")
image = pipe("your prompt", num_inference_steps=24, guidance_scale=2.5).images[0]
image.save("result.png")
```

## Usage (ComfyUI)

1. Download `{lora_filename}` from this repo.
2. Put it into `ComfyUI/models/loras/`.
3. In workflow use LoRA Loader and set this file.

Trigger text: `{trigger_line}`

Generated on {now}.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload a FLUX LoRA adapter to Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True, help="HF repo id, e.g. username/flux2-crack-lora")
    parser.add_argument(
        "--lora-path",
        default="outputs",
        help="Path to .safetensors file or directory containing it (default: outputs)",
    )
    parser.add_argument(
        "--base-model",
        default="black-forest-labs/FLUX.2-dev",
        help="Base model name to write into model card",
    )
    parser.add_argument(
        "--trigger-text",
        default="",
        help="Optional trigger text for prompts (written to model card)",
    )
    parser.add_argument("--private", action="store_true", help="Create private model repo")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without uploading")
    args = parser.parse_args()

    lora_file = _resolve_lora_file(Path(args.lora_path))
    lora_sha256 = _sha256(lora_file)
    adapter_config = lora_file.parent / "adapter_config.json"

    readme_text = _build_readme(
        repo_id=args.repo_id,
        base_model=args.base_model,
        trigger_text=args.trigger_text.strip() or None,
        lora_filename=lora_file.name,
        sha256=lora_sha256,
    )

    print(f"Resolved LoRA: {lora_file}")
    print(f"SHA256: {lora_sha256}")
    print(f"Target repo: {args.repo_id}")
    print(f"Private repo: {args.private}")
    print(f"Will upload: {lora_file.name}, README.md" + (", adapter_config.json" if adapter_config.exists() else ""))

    if args.dry_run:
        print("Dry-run mode, no upload performed.")
        return 0

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        print("Error: Hugging Face token not found. Run `hf auth login` or export HF_TOKEN.", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(lora_file),
        path_in_repo=lora_file.name,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload FLUX LoRA weights",
    )

    if adapter_config.exists():
        api.upload_file(
            path_or_fileobj=str(adapter_config),
            path_in_repo="adapter_config.json",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Upload adapter config",
        )

    api.upload_file(
        path_or_fileobj=readme_text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add model card",
    )

    print(f"Upload complete: https://huggingface.co/{args.repo_id}")
    print(
        "ComfyUI download example:\n"
        f"hf download {args.repo_id} {lora_file.name} --local-dir /path/to/ComfyUI/models/loras"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
