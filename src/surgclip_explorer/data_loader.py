from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


RAW_SUFFIX = "_endo.png"
EXCLUDED_SUFFIXES = (
    "_endo_mask.png",
    "_endo_color_mask.png",
    "_endo_watershed_mask.png",
)


@dataclass(frozen=True)
class LoadedImage:
    image: Image.Image
    name: str
    content_hash: str
    width: int
    height: int


def is_surgical_frame(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(RAW_SUFFIX) and not any(name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)


def discover_frames(dataset_root: Path, limit: int | None = None) -> list[Path]:
    if not dataset_root.is_dir():
        return []
    frames = [path for path in dataset_root.rglob(f"*{RAW_SUFFIX}") if is_surgical_frame(path)]
    frames.sort(key=lambda path: path.relative_to(dataset_root).as_posix())
    return frames if limit is None else frames[: max(0, limit)]


def load_image_bytes(data: bytes, name: str) -> LoadedImage:
    if not data:
        raise ValueError("The selected image is empty.")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.verify()
        with Image.open(io.BytesIO(data)) as opened:
            image = opened.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The selected file is not a valid PNG or JPEG image.") from exc
    width, height = image.size
    return LoadedImage(image=image, name=name, content_hash=hashlib.sha256(data).hexdigest(), width=width, height=height)


def load_image_path(path: Path) -> LoadedImage:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Could not read image: {path}") from exc
    return load_image_bytes(data, path.name)
