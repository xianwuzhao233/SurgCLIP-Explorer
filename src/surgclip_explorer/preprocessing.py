from __future__ import annotations

from PIL import Image


def create_model_view(image: Image.Image, image_size: int = 224) -> Image.Image:
    """Reproduce CLIP's shortest-edge resize followed by a centered square crop."""
    source = image.convert("RGB")
    width, height = source.size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    scale = image_size / min(width, height)
    resized_size = (max(image_size, round(width * scale)), max(image_size, round(height * scale)))
    resized = source.resize(resized_size, Image.Resampling.BICUBIC)
    left = (resized.width - image_size) // 2
    top = (resized.height - image_size) // 2
    return resized.crop((left, top, left + image_size, top + image_size))
