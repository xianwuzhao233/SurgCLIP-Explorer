from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import streamlit as st

from surgclip_explorer.classifiers import FineTunedCLIPAdapter, ZeroShotCLIPAdapter
from surgclip_explorer.config import SURGICAL_CLASSES, load_config
from surgclip_explorer.data_loader import discover_frames, load_image_bytes, load_image_path
from surgclip_explorer.preprocessing import create_model_view
from surgclip_explorer.visualization import overlay_heatmap


st.set_page_config(page_title="SurgCLIP Explorer", page_icon="🔬", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.7rem; padding-bottom: 2rem; max-width: 1450px;}
    [data-testid="stMetric"] {background:#F7FAFB; border:1px solid #E1EAED; padding:0.65rem; border-radius:0.45rem;}
    .research-note {color:#52646B; font-size:0.91rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

CONFIG = load_config()


@st.cache_resource(show_spinner="Loading pretrained CLIP RN50 and encoding prompts…")
def get_zero_shot_adapter(model_name: str, pretrained: str, device: str, cache_dir: str):
    return ZeroShotCLIPAdapter(model_name, pretrained, device, Path(cache_dir))


@st.cache_data(show_spinner=False)
def get_dataset_frames(root: str, limit: int) -> list[str]:
    return [str(path) for path in discover_frames(Path(root), limit=limit)]


def prediction_dataframe(result) -> pd.DataFrame:
    rows = [
        {"class": result.class_names[index], "cosine similarity": float(result.scores[index])}
        for index in reversed(result.ranking)
    ]
    return pd.DataFrame(rows)


if "prediction_cache" not in st.session_state:
    st.session_state.prediction_cache = {}
if "cam_cache" not in st.session_state:
    st.session_state.cam_cache = {}

st.title("SurgCLIP Explorer")
st.caption("Interactive Surgical Scene Recognition · Pretrained CLIP RN50 · Class-specific Grad-CAM")

with st.sidebar:
    st.header("Input & settings")
    input_mode = st.radio("Image source", ["Upload image", "Browse CholecSeg8k"], horizontal=False)
    loaded = None
    if input_mode == "Upload image":
        uploaded = st.file_uploader("PNG or JPEG", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            try:
                loaded = load_image_bytes(uploaded.getvalue(), uploaded.name)
            except ValueError as exc:
                st.error(str(exc))
    else:
        frames = get_dataset_frames(str(CONFIG.dataset_path), CONFIG.max_dataset_frames)
        if not frames:
            st.warning("No CholecSeg8k frames found. Set `SURGCLIP_DATASET_PATH`.")
        else:
            relative_labels = [str(Path(path).relative_to(CONFIG.dataset_path)) for path in frames]
            selected_label = st.selectbox("Dataset frame", relative_labels)
            try:
                loaded = load_image_path(Path(frames[relative_labels.index(selected_label)]))
            except ValueError as exc:
                st.error(str(exc))

    st.divider()
    st.subheader("Model")
    model_choice = st.selectbox("Inference model", ["Zero-shot CLIP RN50"], disabled=True)
    fine_ok, fine_message = FineTunedCLIPAdapter.validate_checkpoint(
        CONFIG.fine_tuned_checkpoint, CONFIG.fine_tuned_metadata
    )
    if fine_ok:
        st.info("Fine-tuned checkpoint is compatible; frontend activation is planned for the comparison phase.")
    else:
        st.caption(f"Fine-tuned CLIP: {fine_message}")

    st.divider()
    st.subheader("Visualization")
    enable_cam = st.toggle("Enable Grad-CAM", value=True)
    target_class = st.selectbox("Target surgical class", SURGICAL_CLASSES, index=1)
    opacity = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05)
    prediction_display = st.radio("Prediction display", ["Top 3", "All classes"], horizontal=True)

if loaded is None:
    st.info("Upload a laparoscopic image or choose the dataset browser to begin real CLIP inference.")
    st.markdown(
        "**What this tool reports:** ranked cosine similarities against 11 predefined surgical class prompts. "
        "Scores are not calibrated probabilities and a high-ranked candidate is not proof that the structure is present."
    )
    st.stop()

interaction_start = time.perf_counter()
st.caption(f"Input: **{loaded.name}** · {loaded.width} × {loaded.height} px · SHA-256 `{loaded.content_hash[:12]}…`")

try:
    adapter = get_zero_shot_adapter(
        CONFIG.model_name, CONFIG.pretrained, CONFIG.device, str(CONFIG.model_cache_dir)
    )
except Exception as exc:
    st.error(f"Could not load CLIP RN50: {exc}")
    st.stop()

prediction_key = (loaded.content_hash, adapter.identity)
prediction_cached = prediction_key in st.session_state.prediction_cache
if prediction_cached:
    prediction = st.session_state.prediction_cache[prediction_key]
else:
    try:
        with st.spinner("Running CLIP image inference…"):
            prediction = adapter.predict(loaded.image)
        st.session_state.prediction_cache[prediction_key] = prediction
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        st.stop()

left, right = st.columns([1.05, 1.0], gap="large")
with left:
    st.subheader("A · Original image")
    st.image(loaded.image, caption=loaded.name, width="stretch")

with right:
    st.subheader("B · Model predictions")
    st.markdown("**Top-3 ranked class candidates**")
    top_rows = []
    for rank, (name, score) in enumerate(prediction.topk(3), start=1):
        top_rows.append({"rank": rank, "class": name, "cosine similarity": f"{score:.4f}"})
    st.dataframe(pd.DataFrame(top_rows).set_index("rank"), width="stretch")
    if prediction_display == "All classes":
        st.dataframe(prediction_dataframe(prediction).iloc[::-1], hide_index=True, width="stretch")
    st.bar_chart(
        prediction_dataframe(prediction),
        x="cosine similarity",
        y="class",
        horizontal=True,
        width="stretch",
    )
    st.caption(f"Grad-CAM target: **{target_class}** · Scores are cosine similarities, not probabilities.")

cam_ms = None
cam_cached = False
if enable_cam:
    st.subheader("C · Class-specific Grad-CAM")
    class_index = SURGICAL_CLASSES.index(target_class)
    cam_key = (loaded.content_hash, adapter.identity, class_index)
    cam_cached = cam_key in st.session_state.cam_cache
    if cam_cached:
        cam_result, cam_ms = st.session_state.cam_cache[cam_key]
    else:
        try:
            start = time.perf_counter()
            with st.spinner(f"Computing gradients for {target_class}…"):
                cam_result = adapter.gradcam(loaded.image, class_index)
            cam_ms = (time.perf_counter() - start) * 1000.0
            st.session_state.cam_cache[cam_key] = (cam_result, cam_ms)
        except Exception as exc:
            st.warning(f"Grad-CAM unavailable for this class/image: {exc}")
            cam_result = None
    if cam_result is not None:
        model_view = create_model_view(loaded.image)
        overlay = overlay_heatmap(model_view, cam_result.heatmap, opacity)
        original_col, overlay_col = st.columns(2, gap="medium")
        original_col.image(model_view, caption="CLIP model-view crop (224 × 224)", width="stretch")
        overlay_col.image(
            overlay,
            caption=f"{target_class} Grad-CAM · opacity {opacity:.2f}",
            width="stretch",
        )
        st.caption(
            "The overlay uses the same center-cropped model view as CLIP, avoiding false alignment with pixels "
            "outside the encoder input. Grad-CAM is diagnostic—not segmentation or causal proof."
        )

st.subheader("D · Model information")
end_to_end_ms = (time.perf_counter() - interaction_start) * 1000.0
metrics = st.columns(5)
metrics[0].metric("Model", f"{CONFIG.model_name} / {CONFIG.pretrained}")
metrics[1].metric("Device", str(adapter.device))
metrics[2].metric("Preprocess", f"{prediction.preprocess_ms:.1f} ms")
metrics[3].metric("Inference", f"{prediction.inference_ms:.1f} ms")
metrics[4].metric("Grad-CAM", "off" if cam_ms is None else f"{cam_ms:.1f} ms")
st.caption(
    f"Model load: {adapter.load_ms:.1f} ms · End-to-end render path: {end_to_end_ms:.1f} ms · "
    f"Prediction: {'cache' if prediction_cached else 'computed'} · "
    f"CAM: {'cache' if cam_cached else ('computed' if cam_ms is not None else 'not computed')}"
)
