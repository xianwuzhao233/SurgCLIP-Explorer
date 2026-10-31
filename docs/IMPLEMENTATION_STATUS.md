# Implementation status

## Implemented

- Uploaded PNG/JPEG validation and optional read-only CholecSeg8k browser.
- Exact original-frame filtering; masks are excluded.
- Real OpenCLIP RN50 `openai` checkpoint loading and local ignored cache.
- Canonical 11-class, five-template prompt ensemble and cosine ranking.
- Top-3, all-class chart/table, model timing, device, and cache reporting.
- Independent class selection and gradient-derived RN50 `layer4[-1]` Grad-CAM.
- Model-view-aligned overlays and immediate opacity adjustment.
- Explicit failure for zero/non-finite CAMs.
- Model/resource and detached result caching.
- Fine-tuned checkpoint/metadata compatibility validator.

## Tested

Actual commands and host-specific outcomes are recorded after verification in this file's **Verification record** section.

## Partially functional

- Fine-tuned checkpoint compatibility can be checked, but fine-tuned prediction and side-by-side UI are intentionally not active.
- MPS selection exists for portable inference, but the current host reports MPS unavailable. CPU is the verified gradient path.

## Future

- Fine-tuned adapter inference, sigmoid thresholds, and same-image side-by-side CAM comparison.
- Formal calibration and broader held-out-video evaluation.

## Verification record

Verified on 2026-10-31 in Conda environment `vlm`:

- Python 3.10.20; PyTorch 2.11.0; OpenCLIP 3.3.0; Streamlit 1.65.0.
- MPS unavailable and CUDA unavailable on this host; selected device: CPU.
- Dataset scan: 8,080 valid CholecSeg8k originals; mask patterns excluded.
- Unit/frontend-initial-state suite: 8 passed. One third-party `anyio` warning concerns a future pytest 10 rename; no project test failed.
- Real model smoke test: official cached OpenAI RN50 loaded, 11 finite scores returned, and Top-3 ranking validated.
- First-process measured timing on the selected frame: model plus prompt load 4,037.0 ms, preprocessing 2.9 ms, image inference 29.5 ms.
- Real liver CAM: target score 0.277491, activation/gradient shape `(1, 2048, 7, 7)`, 65.2 ms.
- Real grasper CAM: target score 0.265969, activation/gradient shape `(1, 2048, 7, 7)`, 62.6 ms.
- Both CAM scalar scores matched the corresponding ordinary inference scores within the smoke-test tolerance; mean absolute map difference was 0.124064.
- Streamlit upload integration via `AppTest`: no exceptions or warnings; prediction, chart, default liver CAM, and timing metrics rendered.
- Live Streamlit server: startup succeeded and `/_stcore/health` returned `ok`.
- Existing parent checkpoint plus exact metadata: compatibility validator returned `(True, "compatible")`; it is not enabled by default or copied into this project.

Timings are one measured run on this host, not real-time performance claims.
