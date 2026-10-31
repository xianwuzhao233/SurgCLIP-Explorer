#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import open_clip
import streamlit
import torch

from surgclip_explorer.config import load_config
from surgclip_explorer.data_loader import discover_frames
from surgclip_explorer.model_loader import select_device


def main() -> None:
    config = load_config()
    print(f"python={platform.python_version()}")
    print(f"executable={sys.executable}")
    print(f"torch={torch.__version__}")
    print(f"open_clip={open_clip.__version__}")
    print(f"streamlit={streamlit.__version__}")
    print(f"mps_available={torch.backends.mps.is_available()}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"selected_device={select_device(config.device)}")
    print(f"dataset={config.dataset_path}")
    print(f"dataset_frames={len(discover_frames(config.dataset_path))}")
    print(f"model_cache={config.model_cache_dir}")


if __name__ == "__main__":
    main()
