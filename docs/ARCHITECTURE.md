# Architecture

## Components

- `app.py`: Streamlit orchestration and graph-free session caches.
- `config.py`: canonical ontology, prompt ensemble, environment-aware paths.
- `data_loader.py`: safe upload decoding and exact CholecSeg8k frame discovery.
- `model_loader.py`: device selection and OpenCLIP RN50 loading.
- `classifiers.py`: common model adapter API, zero-shot implementation, and fine-tuned contract validation.
- `gradcam.py`: target-layer resolution, temporary hook lifecycle, scalar backpropagation, and CAM validation.
- `preprocessing.py`: visible CLIP center-crop geometry.
- `visualization.py`: heatmap colorization and opacity-only blending.

## Model contract

`SurgicalModelAdapter` exposes `predict`, `gradcam`, and a stable model identity. Prediction returns 11 ordered scores plus preprocessing and inference timings. Grad-CAM is a separate gradient-enabled pass so ordinary inference remains under `torch.no_grad()` and no computation graph enters a cache.

The zero-shot adapter uses OpenCLIP RN50 with the OpenAI checkpoint. Each prompt embedding is normalized before averaging and the class mean is normalized again. Image embeddings are normalized, producing unscaled cosine similarities.

## CAM geometry and safety

The RN50 target is resolved dynamically from `model.visual.layer4`; the final block is used when it is sequential. A temporary forward hook stores the activation and attaches a tensor gradient hook. Cleanup occurs in `finally`. Every request creates fresh local state.

The selected text embedding changes the scalar passed to `backward()`. The resulting 7×7 map is resized to 224×224 and displayed on a separately reconstructed shortest-edge-resized, center-cropped model view. Zero-range and non-finite CAMs raise errors.

## Caching

Streamlit resource caching owns the immutable loaded adapter. Session caches contain only detached NumPy prediction/CAM results keyed by image SHA-256, model identity, and (for CAMs) class index. Overlay opacity is applied after CAM retrieval.

## Fine-tuned boundary

The validator requires an RN50 metadata file with the exact 11-class order and a state dictionary containing the parent pipeline's 11×1024 cosine classifier. This makes incompatibility explicit and prevents accidental use of a head whose indices have different meanings.
