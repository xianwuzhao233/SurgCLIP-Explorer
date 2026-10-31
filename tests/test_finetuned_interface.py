import json
from pathlib import Path

from surgclip_explorer.classifiers import FineTunedCLIPAdapter
from surgclip_explorer.config import SURGICAL_CLASSES


def test_missing_checkpoint_is_nonfatal() -> None:
    ok, reason = FineTunedCLIPAdapter.validate_checkpoint(None, None)
    assert not ok
    assert "not configured" in reason


def test_bad_metadata_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not needed because metadata is checked first")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"model_name": "RN50", "target_classes": SURGICAL_CLASSES[::-1]}))
    ok, reason = FineTunedCLIPAdapter.validate_checkpoint(checkpoint, metadata)
    assert not ok
    assert "class names/order" in reason
