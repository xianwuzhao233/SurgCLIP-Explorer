import numpy as np
import torch
from PIL import Image
from torch import nn

from surgclip_explorer.gradcam import compute_gradcam


class TinyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer4 = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        self.projection = nn.Linear(4, 3, bias=False)

    def forward(self, x):
        features = self.layer4(x)
        return self.projection(features.mean((2, 3)))


class TinyCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = TinyVisual()

    def encode_image(self, x):
        return self.visual(x)


def preprocess(_image):
    values = torch.linspace(0.0, 1.0, 3 * 16 * 16)
    return values.reshape(3, 16, 16)


def test_gradcam_uses_selected_target_and_cleans_hook() -> None:
    torch.manual_seed(4)
    model = TinyCLIP().eval()
    image = Image.new("RGB", (16, 16))
    before = len(model.visual.layer4[-1]._forward_hooks)
    first = compute_gradcam(
        model=model,
        preprocess=preprocess,
        image=image,
        text_embedding=torch.tensor([1.0, 0.0, 0.0]),
        device=torch.device("cpu"),
        output_size=(16, 16),
    )
    second = compute_gradcam(
        model=model,
        preprocess=preprocess,
        image=image,
        text_embedding=torch.tensor([0.0, 1.0, 0.0]),
        device=torch.device("cpu"),
        output_size=(16, 16),
    )
    assert first.heatmap.shape == (16, 16)
    assert np.isfinite(first.heatmap).all()
    assert first.target_score != second.target_score
    assert len(model.visual.layer4[-1]._forward_hooks) == before
