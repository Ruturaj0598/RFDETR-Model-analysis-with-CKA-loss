# Deterministic Regularization via Centered Kernel Alignment in RF-DETR

This repository contains the official implementation of **Deterministic Regularization via Centered Kernel Alignment (CKA)** for the **RF-DETR** object detection model. This work introduces a parameter-free, self-supervised regularization term that improves the detection of small and partially occluded objects by enforcing representational consistency across different augmented views of the same image.

##  Overview
Detecting small objects is challenging because their intermediate feature maps often lack sufficient resolution, leading to a weak signal. Our method addresses this by integrating **Centered Kernel Alignment (CKA)** as a deterministic regularizer applied selectively to the hybrid encoder of RF-DETR.

### Key Contributions
*   **Linear CKA Regularization:** A lightweight consistency penalty that aligns feature maps of original and augmented views.
*   **Multi-Layer Hierarchical CKA:** A resolution-aware approach that applies higher weights to earlier, high-resolution layers (C3, C4) to prioritize small object features.
*   **Broad Applicability:** Validated across three diverse domains: industrial X-ray, underwater imagery, and dynamic sports scenarios.

## Architecture
The project builds upon the **RF-DETR** framework, which utilizes a ResNet-50 backbone, a hybrid encoder (AIFI + CCFF), and a 3-layer deformable decoder.

The total training objective is defined as:
$$L_{total} = L_{RF-DETR} + \lambda_{cka} \cdot L_{cka}$$

Where $L_{cka}$ ensures that the model's internal representations remain stable under visual transformations like cropping, flipping, and color jittering.

## 🚀 Key Features
### 1. Linear CKA Variant
Measures similarity between representations in a way that is invariant to orthogonal transformations and scaling.
*   **Efficient:** Minimal computational overhead with no extra learnable parameters.
*   **Centered:** Features are spatially centered before alignment to ensure unbiased similarity scores.

### 2. Multi-Layer (Hierarchical) CKA
Focuses on the network hierarchy to enhance small object detection.
*   **Resolution-Aware Weighting:** Weights are assigned using the formula $w_l = 2.0 - \frac{l}{L-1}$, giving earlier layers (like C3) up to double the influence of later layers.
*   **Progressive Scheduling:** The CKA penalty is ramped from 0 to $\lambda$ over the first 5 epochs to prevent early training instability.

## 📊 Performance Results
The method consistently outperforms the baseline RF-DETR across three major datasets:

| Dataset | Baseline (mAP@50) | With CKA (mAP@50) | Improvement |
| :--- | :---: | :---: | :---: |
| **Al-Cast** (Industrial) | 0.9449 | **0.9565** | +1.16% |
| **Aquatic** (Underwater) | 0.8392 | **0.8654** | +2.62% |
| **Basketball** (Sports) | 0.9274 | **0.9297** | +0.23% |

**Small Object Highlight:** On the Al-Cast dataset, precision for the "Gas-Holes" class (objects < $32^2$ pixels) increased from **0.9308 to 0.9715 (+4.07%)**.

##  Repository Structure
*   `rf-detr/`: Core model architecture and CKA loss modules.
*   `notebooks/`: Comparison and analysis notebooks, including Basketball and Al-Cast evaluations.
*   `experiments/`: Training scripts and configuration files.

---

## Notebooks

Each dataset has three notebooks that must be run in the order shown. All cells are
independent — run one cell at a time to inspect its output before continuing.

### Notebook 1 — Baseline (`*_baseline.ipynb`)

Trains a standard RF-DETR Base model **without any CKA regularization**. This serves
as the comparison reference for all CKA experiments.

| Cell | Purpose | Output |
| :--- | :--- | :--- |
| 0 | Imports & path configuration | Confirmation print |
| 1 | YOLO → COCO annotation conversion | Converted JSON files + counts |
| 2 | GPU info | Device name and VRAM |
| 3 | Model init & training | Per-epoch loss and mAP |
| 4 | Training metrics plot | Loss and mAP curves inline |
| 5 | Load checkpoint & test dataset | Load confirmation |
| 6 | Full test-set evaluation (mAP) | mAP@0.5, mAP@0.5:0.95, mAP@0.75 |
| 7 | Single-image inference | Ground truth vs predictions side by side |

### Notebook 2 — Linear CKA Loss (`*_loss.ipynb`)

Trains RF-DETR with a **single-layer CKA contrastive loss** applied to the encoder
backbone. This is the primary ablation over the baseline.

Before running, confirm that `lwdetr.py` has been patched to expose
`backbone_features` in the model output (Cell 2 verifies this automatically).

| Cell | Purpose | Output |
| :--- | :--- | :--- |
| 0 | Imports & path configuration | Confirmation print |
| 1 | Dataset categories | Class list |
| 2 | Reload modified `lwdetr.py` | Patch verification print |
| 3 | Debug forward pass | `backbone_features` shape at each scale |
| 4 | GPU info | Device name and VRAM |
| 5 | Model init & training with CKA | Per-epoch loss, val loss, mAP, CKA loss |
| 6 | Training metrics plot | RF-DETR metrics image + loss/CKA curves |
| 7 | GPU memory cleanup | Before/after VRAM usage |
| 8 | Load checkpoint & test dataset | Load confirmation |
| 9 | Full test-set evaluation (mAP) | mAP@0.5, mAP@0.5:0.95, mAP@0.75 |
| 10 | Single-image inference | Ground truth vs predictions side by side |

**Key hyperparameter:** `cka_lambda = 0.5` (tune between 0.1 and 1.0).

### Notebook 3 — Multi-Layer CKA (`*_multilayer.ipynb`)

Extends the single-layer variant with **multi-scale CKA** applied across C3, C4, and
C5 encoder layers simultaneously. Includes progressive scheduling and a small-object
focus weighting scheme.

| Cell | Purpose | Output |
| :--- | :--- | :--- |
| 0 | Imports & path configuration | Confirmation print |
| 1 | Dataset categories | Class list |
| 2 | Reload modified `lwdetr.py` | Patch verification print |
| 3 | Debug forward pass | Multi-scale `backbone_features` shapes |
| 4 | GPU info | Device name and VRAM |
| 5 | Model init & training (multi-layer CKA) | Per-epoch loss, val loss, mAP, CKA loss |
| 6 | Training metrics plot | RF-DETR metrics image + loss/CKA curves |
| 7 | GPU memory cleanup | Before/after VRAM usage |
| 8 | Load checkpoint & test dataset | Load confirmation |
| 9 | Full test-set evaluation (mAP) | mAP@0.5, mAP@0.5:0.95, mAP@0.75 |
| 10 | Single-image inference | Ground truth vs predictions side by side |

**Key hyperparameters:**

| Parameter | Value | Description |
| :--- | :---: | :--- |
| `cka_lambda` | 0.5 | Overall regularization weight |
| `cka_augmentation_strength` | `"moderate"` | Strength of augmentations for CKA views |
| `cka_progressive` | `True` | Ramp CKA weight over first 5 epochs |
| `cka_focus_small_objects` | `True` | Upweight C3/C4 for small object sensitivity |
| `cka_temperature` | 1.0 | Softness of alignment; higher = softer |

### Dataset-Specific Notes

| Dataset | Classes | Domain | Annotation format |
| :--- | :---: | :--- | :--- |
| **Aquatic** | 7 | Underwater animals | YOLO → COCO (converted in baseline Cell 1) |
| **Al-Cast** | 2 | Industrial X-ray defects | YOLO → COCO (converted in baseline Cell 1) |
| **Basketball** | 6 | Sports action detection | YOLO → COCO (converted in baseline Cell 1) |

---

## Setup

### Prerequisites

```bash
pip install rfdetr supervision pillow tqdm pandas matplotlib
```

### Configuration

Before running any notebook, set the two path variables in **Cell 0**:

```python
DATASET_ROOT = Path("path/to/dataset")   # contains train/ valid/ test/
OUTPUT_DIR   = Path("path/to/output")    # checkpoints written here
```

Set your Roboflow API key as an environment variable — **do not hardcode it** in the
notebook:

```bash
# Windows PowerShell
$env:ROBOFLOW_API_KEY = "your_key_here"

# Linux / macOS
export ROBOFLOW_API_KEY="your_key_here"
```

Then read it in the notebook:

```python
import os
api_key = os.environ.get("ROBOFLOW_API_KEY", "")
```

### Default CKA Hyperparameters (from the paper)

| Parameter | Value |
| :--- | :---: |
| `cka_lambda` | 0.5 |
| `cka_temperature` | 1.0 |
| Layer weights (C3, C4, C5) | 2.0, 1.5, 1.0 |

---

## Modified Files

CKA regularization is introduced through **three targeted changes** to the
standard RF-DETR codebase. Everything else — the decoder, criterion, and
optimiser — is left untouched.

| File | What was changed |
| :--- | :--- |
| `rfdetr/models/lwdetr.py` | `LWDETR.forward()` extended to collect and return intermediate backbone feature maps alongside the standard detection outputs |
| `rfdetr/engine.py` | `train_one_epoch()` extended to run a second augmented forward pass and compute the CKA loss on the collected features; helper functions added above the training loop |
| `rfdetr/config.py` | Two new fields added to `TrainConfig`: `cka_lambda` (default `0.0`, disabling CKA) and `cka_scales` |

Setting `cka_lambda = 0.0` reproduces the original RF-DETR baseline exactly — the
second forward pass is never triggered and no CKA objects are instantiated.

---

## Installation

### Option A — Clone this repository (recommended)

This repository already contains the modified RF-DETR source. Install it directly:

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo

pip install -r requirements.txt
# or, if an editable install is needed:
pip install -e .
```

### Option B — Patch an existing RF-DETR installation

If you already have a working RF-DETR environment and only want to apply the
CKA modifications, replace the three files manually:

```bash
# 1. Find where RF-DETR is installed
python -c "import rfdetr; print(rfdetr.__file__)"
# Example output: /path/to/site-packages/rfdetr/__init__.py

# 2. Copy the three modified files into that location
cp rfdetr/models/lwdetr.py  /path/to/site-packages/rfdetr/models/lwdetr.py
cp rfdetr/engine.py         /path/to/site-packages/rfdetr/engine.py
cp rfdetr/config.py         /path/to/site-packages/rfdetr/config.py
```

> ⚠️ **Back up the originals first** if you want to restore the unmodified baseline:
> ```bash
> cp /path/to/site-packages/rfdetr/models/lwdetr.py lwdetr.py.bak
> cp /path/to/site-packages/rfdetr/engine.py        engine.py.bak
> cp /path/to/site-packages/rfdetr/config.py        config.py.bak
> ```

### Verify the patch is active

Run this in a Python terminal or notebook cell after installation:

```python
import inspect
import rfdetr.models.lwdetr

source = inspect.getsource(rfdetr.models.lwdetr.LWDETR.forward)
print("✓ Patch active" if "backbone_features" in source else "✗ Patch NOT found")
```

This is the same check run automatically in **Cell 2** of every notebook.

##  Citation
If you find this work useful for your research, please cite:
```bibtex
@inproceedings{madan2026deterministic,
  title={Deterministic Regularization via Centered Kernel Alignment in RF-DETR},
  author={Madan, Manav and Shelake, Ruturaj and Reich, Christoph},
  booktitle={IEEE ETAACT-2026},
  year={2026}
}
```

## 🙏Acknowledgements
This work was funded by the **DFG** (grant RE 2881/6-1) and the **ANR** (grant ANR-22-CE92-0007). This research was conducted at **IDACUS, Furtwangen University**.
