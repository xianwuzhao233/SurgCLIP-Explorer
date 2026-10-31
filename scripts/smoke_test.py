#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from surgclip_explorer.classifiers import ZeroShotCLIPAdapter
from surgclip_explorer.config import SURGICAL_CLASSES, load_config
from surgclip_explorer.data_loader import discover_frames, load_image_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real RN50 inference and optional two-class Grad-CAM checks.")
    parser.add_argument("--with-gradcam", action="store_true")
    args = parser.parse_args()
    config = load_config()
    frames = discover_frames(config.dataset_path, limit=1)
    if not frames:
        raise SystemExit(f"No dataset frame found under {config.dataset_path}")
    loaded = load_image_path(frames[0])
    start = time.perf_counter()
    adapter = ZeroShotCLIPAdapter(
        config.model_name, config.pretrained, config.device, config.model_cache_dir
    )
    prediction = adapter.predict(loaded.image)
    assert prediction.scores.shape == (11,)
    assert np.isfinite(prediction.scores).all()
    assert list(prediction.ranking) == list(np.argsort(-prediction.scores))
    print(f"model={adapter.identity}")
    print(f"device={adapter.device}")
    print(f"image={frames[0]}")
    print(f"scores={len(prediction.scores)} finite=true")
    print(f"top3={prediction.topk(3)}")
    print(f"load_ms={adapter.load_ms:.1f} preprocess_ms={prediction.preprocess_ms:.1f} inference_ms={prediction.inference_ms:.1f}")
    if args.with_gradcam:
        results = {}
        for name in ("liver", "grasper"):
            index = SURGICAL_CLASSES.index(name)
            cam_start = time.perf_counter()
            result = adapter.gradcam(loaded.image, index)
            assert result.heatmap.shape == (224, 224)
            assert np.isfinite(result.heatmap).all()
            assert result.activation_shape[-2:] == result.gradient_shape[-2:]
            assert abs(result.target_score - float(prediction.scores[index])) < 5e-4
            results[name] = result.heatmap
            print(
                f"cam={name} score={result.target_score:.6f} shape={result.heatmap.shape} "
                f"activation={result.activation_shape} gradient={result.gradient_shape} "
                f"latency_ms={(time.perf_counter() - cam_start) * 1000:.1f}"
            )
        print(f"cam_mean_absolute_difference={np.mean(np.abs(results['liver'] - results['grasper'])):.6f}")
    print(f"total_ms={(time.perf_counter() - start) * 1000:.1f}")


if __name__ == "__main__":
    main()
