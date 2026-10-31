# Two-minute demo guide

## Startup

```bash
conda activate vlm
cd SurgCLIP-Explorer
streamlit run app.py
```

Before recording, let the pretrained RN50 weight download complete once. Use a CholecSeg8k frame with both liver and a grasper visibly present. Do not present rankings as ground-truth detections.

## Sequence and narration

### 0:00–0:20 — Startup

Show the terminal command and browser load.

> This is SurgCLIP Explorer, a Streamlit prototype using the real OpenAI-pretrained CLIP RN50 through OpenCLIP. The model and prompt embeddings are cached, while image predictions and explanations are keyed by image content.

### 0:20–0:45 — Image recognition

Choose **Browse CholecSeg8k** or upload the prepared frame. Point to its dimensions, Top-3, and all-class chart.

> The system compares the frame with five surgical prompts for each of eleven structures and instruments. These are cosine similarities and ranked candidates, not calibrated probabilities or proof of presence.

### 0:45–1:15 — Interactive Grad-CAM

Choose **liver**, then **grasper**. Wait for each computed CAM.

> Selecting liver backpropagates the liver cosine score to RN50's final spatial convolutional stage. Switching to grasper performs a new backward pass with the grasper score. The overlay is aligned to CLIP's actual center-cropped model input. Grad-CAM is diagnostic, not segmentation.

### 1:15–1:40 — Input manipulation

Select a second frame. Show changed ranking and CAM; move opacity to demonstrate immediate reblending.

> The image hash changes, so inference and CAM are recomputed. Opacity is presentation-only and does not rerun the network.

### 1:40–2:00 — Future extension

Point to the sidebar fine-tuned status and model information.

> Zero-shot inference and class-specific Grad-CAM are complete. A compatibility validator already checks the existing cosine-head checkpoint shape and exact class order. The future phase will activate a fair side-by-side fine-tuned comparison; it is not presented as working today.

## Troubleshooting

- If the model is still downloading, stop the recording and retry after it finishes.
- If a class produces a degenerate CAM, choose another real frame; the app correctly refuses to fabricate a heatmap.
- Keep the browser at a width that preserves the two-column Grad-CAM comparison.
