# SurgCLIP Explorer

**Interactive Surgical Scene Recognition and Class-Specific Grad-CAM Analysis**

SurgCLIP Explorer is a research-oriented Streamlit application for examining pretrained CLIP RN50 predictions on laparoscopic images. The verified baseline performs image-based analysis over 11 predefined surgical structures and instruments and provides class-specific visual attribution through Grad-CAM.

The broader research goal is to study how zero-shot vision-language representations differ from surgically adapted CLIP models. The current release prioritizes a reproducible zero-shot baseline; it does not claim that a fine-tuned comparison is already operational.

> [!CAUTION]
> This software is a research prototype. It is not a medical device and must not be used for clinical diagnosis, treatment, or decision-making.

## Key features

Implemented and verified:

- PNG and JPEG image upload without requiring a local dataset.
- Optional read-only CholecSeg8k frame browser using an external path.
- Real OpenCLIP RN50 inference with the OpenAI pretrained checkpoint.
- An ensemble of five surgical text prompts for each of 11 target classes.
- Top-3 candidates and all-class cosine-similarity visualization.
- Independent target-class selection for gradient-based RN50 Grad-CAM.
- Heatmap opacity controls that do not rerun model inference.
- Model, prediction, and class-aware result caching.
- Explicit validation of a future fine-tuned checkpoint contract.

Planned, not implemented in the current frontend:

- Fine-tuned CLIP inference and calibrated multilabel thresholds.
- Side-by-side zero-shot and surgically adapted predictions and CAMs.

## System architecture

```text
User image
    │
    ▼
OpenCLIP RN50 preprocessing
    │
    ▼
CLIP RN50 image encoder ── cosine comparison with prompt ensemble
    │                                      │
    │                                      ▼
    │                              11 ranked class scores
    │
    └── selected class scalar ── backward gradients ── layer4[-1] Grad-CAM
                                                   │
                                                   ▼
                                      Streamlit visualization
```

Ordinary inference and gradient-enabled attribution are separate operations. Only detached results are cached. Grad-CAM is displayed on the exact 224×224 center-cropped model view so the overlay is not stretched over pixels that CLIP did not observe.

The common model boundary is `SurgicalModelAdapter`. `ZeroShotCLIPAdapter` is fully functional; `FineTunedCLIPAdapter` currently validates backbone, embedding dimension, class ordering, head structure, state keys, and optional thresholds before any future integration is enabled.

## Requirements

- Python 3.10.20
- PyTorch 2.11.0
- `open_clip_torch` 3.3.0
- Streamlit 1.65.0
- Network access on the first model load, unless the official RN50 checkpoint is already cached

The baseline was verified on CPU. CUDA remains supported through PyTorch. The shared adapter conservatively falls back to CPU when MPS is selected because CPU is the verified Grad-CAM path. RN50 inference and attribution do not require a discrete GPU, but CPU latency varies by machine.

## Installation

### Option A: Conda environment

```bash
git clone <repository-url>
cd SurgCLIP-Explorer
conda env create -f environment.yml
conda activate surgclip-explorer
```

### Option B: Existing Python 3.10 environment

```bash
git clone <repository-url>
cd SurgCLIP-Explorer
python -m pip install -r requirements.txt
```

The dependency list is intentionally limited to packages used by the application and tests; it is not an export of the larger research environment.

## Quick start

No dataset is needed for the upload workflow:

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit and upload a PNG or JPEG laparoscopic frame.

## Dataset configuration

CholecSeg8k is optional and must remain outside this repository. To enable the frame browser:

```bash
export SURGCLIP_DATASET_PATH=/absolute/path/to/CholecSeg8k
streamlit run app.py
```

The default relative lookup is `../Data/CholecSeg8k`. A missing dataset path does not block image upload. The browser recursively includes only original `frame_*_endo.png` files and excludes ordinary, color, and watershed masks. It never writes into the dataset.

## Model weights

Model weights are not distributed in this repository. On first use, OpenCLIP downloads the official OpenAI RN50 checkpoint into the ignored `model_cache/` directory. Later runs reuse that cache. If an official legacy `RN50.pt` is already placed in `model_cache/`, the loader uses it directly with the OpenAI preprocessing metadata.

The first download requires network access and may take several minutes. The application never substitutes random or mock weights when loading fails.

## Usage

1. Start the application with `streamlit run app.py`.
2. Upload a laparoscopic PNG/JPEG, or configure and select a CholecSeg8k frame.
3. Inspect the Top-3 ranked candidates and all 11 cosine similarities.
4. Select a surgical target class in the sidebar.
5. Inspect its class-specific Grad-CAM overlay.
6. Select another class to trigger a new gradient target and compare the visualization.
7. Adjust opacity to reblend the existing CAM without rerunning the network.

CLIP values are reported as cosine similarities, not probabilities. A Top-3 rank does not establish that all three structures are present.

## Reproducibility and tests

Run the automated suite:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest -q
```

Inspect the environment and external dataset discovery:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/check_environment.py
```

Run real pretrained-model verification on one externally configured CholecSeg8k frame:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/smoke_test.py
KMP_DUPLICATE_LIB_OK=TRUE python scripts/smoke_test.py --with-gradcam
```

The smoke test checks real RN50 weight loading, exactly 11 finite scores, ordering, and independent liver/grasper gradient passes. It does not copy the selected image into the repository.

The verified development environment contains duplicate OpenMP runtimes, so the scripts set or document `KMP_DUPLICATE_LIB_OK=TRUE`, matching the parent research workflow. A clean environment containing one OpenMP runtime is preferable for extended experiments.

## Future fine-tuning integration

The existing research checkpoint format uses a frozen RN50 encoder and a text-initialized cosine multilabel head with 11×1024 class weights, per-class logits, sigmoid probabilities, and optional validation thresholds. Integration is gated by `FineTunedCLIPAdapter.validate_checkpoint` and an exact metadata file; see `configs/finetuned_metadata.example.json`.

The current UI does not load this adapter or claim fine-tuned results. A future release will use the same image and class target for a fair side-by-side comparison.

## Limitations

- CLIP cosine similarity is not a calibrated probability.
- Top-ranked classes are not guaranteed to be present in the image.
- The fixed ontology does not support open-ended surgical questions.
- RN50's final 7×7 spatial feature map limits CAM resolution.
- Center-crop preprocessing may omit structures near image boundaries.
- Grad-CAM is a diagnostic visualization, not a segmentation mask or causal explanation.
- Performance and failure modes have not been validated for clinical use.

## Project layout

```text
SurgCLIP-Explorer/
├── app.py                         # Streamlit entry point
├── configs/                       # Portable application and metadata configuration
├── src/surgclip_explorer/         # Inference, data, Grad-CAM, and visualization modules
├── scripts/                       # Environment and real-model smoke checks
├── tests/                         # Unit and Streamlit integration tests
├── docs/                          # Architecture, status, and demonstration guide
├── requirements.txt               # Verified minimal Python dependencies
└── environment.yml                # Reproducible Python 3.10 Conda environment
```

## License and citation

No software license has been selected for this repository. No `LICENSE` file is included, and reuse permissions should not be inferred. Third-party packages and pretrained weights remain subject to their own licenses and terms.

No DOI or archival software citation is available for this baseline. If citing the project in coursework, reference the repository URL and the commit SHA used for the experiment.
