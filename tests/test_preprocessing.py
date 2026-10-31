from PIL import Image

from surgclip_explorer.preprocessing import create_model_view


def test_model_view_preserves_clip_crop_geometry() -> None:
    image = Image.new("RGB", (640, 480))
    assert create_model_view(image).size == (224, 224)
