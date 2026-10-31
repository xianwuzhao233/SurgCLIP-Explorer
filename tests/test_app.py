from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_initial_state_renders_without_loading_model() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=15).run()
    assert not app.exception
    assert app.title[0].value == "SurgCLIP Explorer"
    assert any("Upload a laparoscopic image" in message.value for message in app.info)
