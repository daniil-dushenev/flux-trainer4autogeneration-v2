from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_images(images_dir: Path) -> Iterable[Path]:
    for p in sorted(images_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def validate_images(images_dir: Path) -> None:
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    imgs = list(iter_images(images_dir))
    if not imgs:
        raise ValueError(f"no images found in {images_dir}")
    for p in imgs:
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as exc:
            raise ValueError(f"invalid image: {p}") from exc
