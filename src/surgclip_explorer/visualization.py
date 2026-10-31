from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from PIL import Image


def colorize_heatmap(heatmap: np.ndarray) -> Image.Image:
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
        raise ValueError("Heatmap must be a finite 2D array.")
    rgba = colormaps["turbo"](np.clip(heatmap, 0.0, 1.0))
    return Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8), mode="RGB")


def overlay_heatmap(model_view: Image.Image, heatmap: np.ndarray, opacity: float) -> Image.Image:
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("Opacity must be between 0 and 1.")
    base = model_view.convert("RGB")
    colored = colorize_heatmap(heatmap).resize(base.size, Image.Resampling.BILINEAR)
    return Image.blend(base, colored, opacity)
