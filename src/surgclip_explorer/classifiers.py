from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .config import PROMPT_TEMPLATES, SURGICAL_CLASSES
from .gradcam import GradCAMResult, compute_gradcam
from .model_loader import load_openclip_rn50, select_device


@dataclass(frozen=True)
class PredictionResult:
    class_names: tuple[str, ...]
    scores: np.ndarray
    ranking: tuple[int, ...]
    preprocess_ms: float
    inference_ms: float
    output_kind: str

    def topk(self, k: int = 3) -> list[tuple[str, float]]:
        return [(self.class_names[index], float(self.scores[index])) for index in self.ranking[:k]]


class SurgicalModelAdapter(ABC):
    class_names = tuple(SURGICAL_CLASSES)

    @abstractmethod
    def predict(self, image: Image.Image) -> PredictionResult: ...

    @abstractmethod
    def gradcam(self, image: Image.Image, class_index: int) -> GradCAMResult: ...

    @property
    @abstractmethod
    def identity(self) -> str: ...


class ZeroShotCLIPAdapter(SurgicalModelAdapter):
    def __init__(self, model_name: str, pretrained: str, requested_device: str, cache_dir: Path) -> None:
        start = time.perf_counter()
        self.device = select_device(requested_device)
        self.gradcam_device = self.device
        if self.device.type == "mps":
            # CPU is the conservative, verified gradient backend for this application.
            self.gradcam_device = torch.device("cpu")
            self.device = torch.device("cpu")
        self.model, self.preprocess, self.tokenizer = load_openclip_rn50(
            model_name, pretrained, self.device, cache_dir
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model_name = model_name
        self.pretrained = pretrained
        self.text_embeddings = self._build_text_embeddings()
        self.load_ms = (time.perf_counter() - start) * 1000.0

    @property
    def identity(self) -> str:
        return f"open_clip:{self.model_name}:{self.pretrained}:zero-shot"

    def _build_text_embeddings(self) -> torch.Tensor:
        class_embeddings = []
        with torch.no_grad():
            for class_name in self.class_names:
                prompts = [template.format(class_name) for template in PROMPT_TEMPLATES]
                tokens = self.tokenizer(prompts).to(self.device)
                prompt_features = F.normalize(self.model.encode_text(tokens).float(), dim=-1)
                class_embeddings.append(F.normalize(prompt_features.mean(dim=0), dim=-1))
        return torch.stack(class_embeddings, dim=0)

    def predict(self, image: Image.Image) -> PredictionResult:
        start = time.perf_counter()
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        preprocess_ms = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        with torch.no_grad():
            image_features = F.normalize(self.model.encode_image(tensor).float(), dim=-1)
            scores_tensor = image_features @ self.text_embeddings.T
        inference_ms = (time.perf_counter() - start) * 1000.0
        scores = scores_tensor[0].detach().cpu().numpy().astype(np.float32)
        ranking = tuple(np.argsort(-scores).tolist())
        return PredictionResult(self.class_names, scores, ranking, preprocess_ms, inference_ms, "cosine similarity")

    def gradcam(self, image: Image.Image, class_index: int) -> GradCAMResult:
        if not 0 <= class_index < len(self.class_names):
            raise IndexError("Grad-CAM class index is out of range.")
        return compute_gradcam(
            model=self.model,
            preprocess=self.preprocess,
            image=image,
            text_embedding=self.text_embeddings[class_index],
            device=self.gradcam_device,
        )


class CosineMultilabelHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.07))))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        return scale * F.linear(features, F.normalize(self.weight, dim=-1), self.bias)


class FineTunedCLIPAdapter(SurgicalModelAdapter):
    """Loader contract for the existing text-initialized cosine-head checkpoints."""

    @staticmethod
    def validate_checkpoint(checkpoint: Path | None, metadata: Path | None) -> tuple[bool, str]:
        if checkpoint is None:
            return False, "checkpoint not configured"
        if not checkpoint.is_file():
            return False, f"checkpoint not found: {checkpoint}"
        if metadata is None or not metadata.is_file():
            return False, "class-order metadata is required for compatibility validation"
        try:
            details = json.loads(metadata.read_text(encoding="utf-8"))
        except Exception as exc:
            return False, f"invalid metadata JSON: {exc}"
        if details.get("model_name") != "RN50":
            return False, "metadata backbone must be RN50"
        if details.get("target_classes") != SURGICAL_CLASSES:
            return False, "metadata class names/order do not match the canonical ontology"
        if details.get("visual_embedding_dim") != 1024:
            return False, "metadata visual_embedding_dim must be 1024"
        if details.get("head") != "CosineMultilabelHead":
            return False, "metadata head must be CosineMultilabelHead"
        if details.get("pretrained") != "openai":
            return False, "metadata pretrained identifier must be openai"
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = payload.get("model_state_dict", {})
            if tuple(state["classifier.weight"].shape) != (len(SURGICAL_CLASSES), 1024):
                return False, "classifier.weight must have shape (11, 1024)"
            for key in ("classifier.bias", "classifier.logit_scale"):
                if key not in state:
                    return False, f"missing state key: {key}"
            if "clip_model.visual.conv1.weight" not in state:
                return False, "checkpoint does not contain the RN50 visual encoder"
            thresholds = payload.get("decision_thresholds", details.get("decision_thresholds"))
            if thresholds is not None:
                if not isinstance(thresholds, list) or len(thresholds) != len(SURGICAL_CLASSES):
                    return False, "decision_thresholds must contain exactly 11 values"
                if not all(0.0 <= float(value) <= 1.0 for value in thresholds):
                    return False, "decision_thresholds values must be in [0, 1]"
        except Exception as exc:
            return False, f"checkpoint could not be inspected: {exc}"
        return True, "compatible"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "The fine-tuned frontend is intentionally gated until a checkpoint and exact metadata are configured."
        )

    @property
    def identity(self) -> str:
        return "fine-tuned:unloaded"

    def predict(self, image: Image.Image) -> PredictionResult:
        raise NotImplementedError

    def gradcam(self, image: Image.Image, class_index: int) -> GradCAMResult:
        raise NotImplementedError
