from __future__ import annotations

import os
from pathlib import Path

import torch


def select_device(requested: str = "auto", *, for_gradcam: bool = False) -> torch.device:
    normalized = requested.strip().lower()
    if normalized not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    if normalized == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and not for_gradcam:
        return torch.device("mps")
    return torch.device("cpu")


def load_openclip_rn50(model_name: str, pretrained: str, device: torch.device, cache_dir: Path):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError("open_clip_torch is required. Install the project requirements.") from exc
    # open_clip>=3 may prefer the Hugging Face mirror for a tag even when the
    # original OpenAI .pt file is already cached. Use that verified local file
    # explicitly and carry over the tag's preprocessing metadata.
    pretrained_value = pretrained
    transform_overrides = {}
    local_openai_checkpoint = cache_dir / f"{model_name}.pt"
    if pretrained == "openai" and local_openai_checkpoint.is_file():
        pretrained_value = str(local_openai_checkpoint)
        pretrained_cfg = open_clip.get_pretrained_cfg(model_name, pretrained)
        transform_overrides = {
            "image_mean": pretrained_cfg.get("mean"),
            "image_std": pretrained_cfg.get("std"),
            "image_interpolation": pretrained_cfg.get("interpolation"),
            "image_resize_mode": pretrained_cfg.get("resize_mode"),
        }
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_value,
        device=device,
        cache_dir=str(cache_dir),
        # The official OpenAI RN50.pt cache is a TorchScript archive. PyTorch
        # 2.6+ requires this explicit opt-out for trusted legacy archives.
        weights_only=not local_openai_checkpoint.is_file(),
        **transform_overrides,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer
