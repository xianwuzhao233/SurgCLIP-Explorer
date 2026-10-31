from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SURGICAL_CLASSES = [
    "abdominal wall",
    "liver",
    "gastrointestinal tract",
    "fat",
    "grasper",
    "connective tissue",
    "cystic duct",
    "L-hook electrocautery",
    "gallbladder",
    "hepatic vein",
    "liver ligament",
]

PROMPT_TEMPLATES = [
    "a laparoscopic image showing {}",
    "a surgical endoscopic image containing {}",
    "a cholecystectomy scene with {}",
    "an endoscopic surgery frame where {} is visible",
    "a surgical scene containing {}",
]


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    dataset_path: Path
    model_name: str
    pretrained: str
    device: str
    model_cache_dir: Path
    max_dataset_frames: int
    fine_tuned_checkpoint: Path | None
    fine_tuned_metadata: Path | None


def _resolve_path(value: str | None, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def load_config(path: Path | None = None) -> AppConfig:
    project_root = Path(__file__).resolve().parents[2]
    config_path = path or project_root / "configs" / "default.yaml"
    raw: dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw = loaded

    dataset_value = os.getenv("SURGCLIP_DATASET_PATH", raw.get("dataset_path", "../Data/CholecSeg8k"))
    cache_value = os.getenv("SURGCLIP_MODEL_CACHE", raw.get("model_cache_dir", "model_cache"))
    checkpoint_value = os.getenv("SURGCLIP_FINETUNED_CHECKPOINT", raw.get("fine_tuned_checkpoint"))
    metadata_value = os.getenv("SURGCLIP_FINETUNED_METADATA", raw.get("fine_tuned_metadata"))

    dataset_path = _resolve_path(str(dataset_value), project_root)
    model_cache_dir = _resolve_path(str(cache_value), project_root)
    assert dataset_path is not None and model_cache_dir is not None
    return AppConfig(
        project_root=project_root,
        dataset_path=dataset_path,
        model_name=str(raw.get("model_name", "RN50")),
        pretrained=str(raw.get("pretrained", "openai")),
        device=os.getenv("SURGCLIP_DEVICE", str(raw.get("device", "auto"))),
        model_cache_dir=model_cache_dir,
        max_dataset_frames=int(raw.get("max_dataset_frames", 500)),
        fine_tuned_checkpoint=_resolve_path(checkpoint_value, project_root),
        fine_tuned_metadata=_resolve_path(metadata_value, project_root),
    )
