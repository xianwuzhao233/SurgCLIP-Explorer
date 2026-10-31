from pathlib import Path

from PIL import Image

from surgclip_explorer.data_loader import discover_frames, is_surgical_frame, load_image_path


def test_frame_filter_excludes_all_masks(tmp_path: Path) -> None:
    names = [
        "frame_1_endo.png",
        "frame_1_endo_mask.png",
        "frame_1_endo_color_mask.png",
        "frame_1_endo_watershed_mask.png",
    ]
    for name in names:
        Image.new("RGB", (10, 8)).save(tmp_path / name)
    frames = discover_frames(tmp_path)
    assert [path.name for path in frames] == ["frame_1_endo.png"]
    assert is_surgical_frame(frames[0])


def test_load_image_converts_to_rgb(tmp_path: Path) -> None:
    path = tmp_path / "frame_2_endo.png"
    Image.new("L", (13, 7), color=100).save(path)
    loaded = load_image_path(path)
    assert loaded.image.mode == "RGB"
    assert (loaded.width, loaded.height) == (13, 7)
    assert len(loaded.content_hash) == 64
