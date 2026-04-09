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

## 🛠 Usage
### Configuration
The default CKA hyperparameters used in the paper are:
*   `cka_lambda`: 0.5
*   `temperature (τ)`: 1.0
*   `layers`: [C3, C4, C5] with weights [2.0, 1.5, 1.0]

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
