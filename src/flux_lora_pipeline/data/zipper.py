from __future__ import annotations

from pathlib import Path
import zipfile

from flux_lora_pipeline.data.dataset import iter_images


def make_dataset_zip(
    images_dir: Path,
    captions_dir: Path | None,
    out_path: Path,
    default_caption: str,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for img_path in iter_images(images_dir):
            zf.write(img_path, arcname=img_path.name)
            if captions_dir:
                cap_path = captions_dir / (img_path.stem + ".txt")
                if cap_path.exists():
                    zf.write(cap_path, arcname=cap_path.name)
                    continue
            # fallback caption
            caption_name = f"{img_path.stem}.txt"
            zf.writestr(caption_name, default_caption)

    return out_path
