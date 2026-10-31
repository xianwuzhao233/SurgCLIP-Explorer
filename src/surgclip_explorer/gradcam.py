from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class DegenerateCAMError(RuntimeError):
    pass


@dataclass(frozen=True)
class GradCAMResult:
    heatmap: np.ndarray
    target_score: float
    activation_shape: tuple[int, ...]
    gradient_shape: tuple[int, ...]


def resolve_rn50_target_layer(model: Any) -> torch.nn.Module:
    visual = getattr(model, "visual", None)
    layer4 = getattr(visual, "layer4", None)
    if layer4 is None:
        raise RuntimeError("The visual encoder is not a compatible CLIP RN50 (visual.layer4 missing).")
    if isinstance(layer4, torch.nn.Sequential) and len(layer4):
        return layer4[-1]
    return layer4


def compute_gradcam(
    *,
    model: Any,
    preprocess: Any,
    image: Any,
    text_embedding: torch.Tensor,
    device: torch.device,
    output_size: tuple[int, int] = (224, 224),
) -> GradCAMResult:
    activation: torch.Tensor | None = None
    gradient: torch.Tensor | None = None
    target_layer = resolve_rn50_target_layer(model)

    def capture_activation(_module, _inputs, output):
        nonlocal activation, gradient
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("Grad-CAM target layer did not return a tensor.")
        activation = output
        if output.requires_grad:
            output.register_hook(lambda value: _store_gradient(value))

    def _store_gradient(value: torch.Tensor) -> None:
        nonlocal gradient
        gradient = value

    handle = target_layer.register_forward_hook(capture_activation)
    try:
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        image_tensor.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        image_features = model.encode_image(image_tensor)
        image_features = F.normalize(image_features.float(), dim=-1)
        target = F.normalize(text_embedding.to(device=device, dtype=torch.float32), dim=-1)
        score = (image_features * target.unsqueeze(0)).sum()
        score.backward()
    finally:
        handle.remove()

    if activation is None or gradient is None:
        raise RuntimeError("Grad-CAM hooks did not capture both activations and gradients.")
    if activation.ndim != 4 or gradient.shape != activation.shape:
        raise RuntimeError(
            f"Unexpected Grad-CAM tensors: activation={tuple(activation.shape)}, gradient={tuple(gradient.shape)}"
        )
    weights = gradient.detach().float().mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation.detach().float()).sum(dim=1, keepdim=True))
    cam_min, cam_max = cam.min(), cam.max()
    dynamic_range = float((cam_max - cam_min).item())
    if not np.isfinite(dynamic_range) or dynamic_range <= 1e-8:
        raise DegenerateCAMError("The selected class produced a zero or degenerate Grad-CAM map.")
    cam = (cam - cam_min) / (cam_max - cam_min)
    cam = F.interpolate(cam, size=(output_size[1], output_size[0]), mode="bilinear", align_corners=False)
    heatmap = cam[0, 0].detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(heatmap).all():
        raise DegenerateCAMError("The Grad-CAM map contains non-finite values.")
    return GradCAMResult(
        heatmap=heatmap,
        target_score=float(score.detach().cpu().item()),
        activation_shape=tuple(activation.shape),
        gradient_shape=tuple(gradient.shape),
    )
