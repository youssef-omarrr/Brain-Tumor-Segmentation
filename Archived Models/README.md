# Archived Models: Brain Tumor Segmentation

This document summarizes the three archived model versions (`v0`, `v1`, `v2`) that preceded the current pipeline. It explains architecture differences, training choices, measured results, and the lessons learned from each iteration. Use this as a quick reference when inspecting old experiments or deciding what to carry forward into `v3`.

> These archived models were developed early in my workflow before I had experience in organizing code and experiments. 
> 
> Each version is implemented as a **single notebook** that contains dataset creation, model definition, training loop, and evaluation all in one place, useful for quick experiments but not modular or reproducible. 
> 
> For **v3** I plan to reorganize the project into clear folders and separate modules and split responsibilities (dataset, model, training, utils) so the codebase is easier to maintain, reproduce, and extend.

---

## TLDR: which is which

- **v0**: baseline experiments: vanilla **U-Net**, basic pipeline, established data / notebook structure.
- **v1**: transfer-learning experiments: **DeepLabV3 (ResNet-50)** adapted for grayscale input, more stable training tools.
- **v2**: best archived model: **UNet++ with pretrained EfficientNet-B4 encoder**, deep supervision, advanced loss and augmentations.

---
## Version-by-version breakdown

### v0: Baseline (vanilla U-Net)

#### **Architecture**

* Classic U-Net encoder–decoder with 4 down/up blocks.
* Standard conv → ReLU → BatchNorm pattern.

#### **Training / pipeline**

* Simple preprocessing and manual normalization.
* Loss: Binary Cross Entropy (BCE).
* Short experiment runs used to establish pipeline and data alignment.

#### **Results**

* Final Validation Accuracy: **90.81%** (pixel-wise accuracy).
* Dice: **not reported** / unavailable due to earlier metric implementation issues.

#### **Notes / Lessons**

* **Pixel-wise accuracy is misleading for this imbalanced task**: the model could get high accuracy by just predicting the background and missing the tumor (because the rumor maybe just a few number of pixels).
* Established the data pairing, resizing, and mask binarization logic used by later versions.
* Demonstrated the need for better *loss functions* and more robust *augmentations* when tumors occupy a small image area.

---

### v1: Transfer-learning (DeepLabV3, ResNet-50)

#### **Architecture**

* `torchvision`'s DeepLabV3 with a ResNet-50 backbone (pretrained).
* First conv adjusted to accept single-channel MRI slices.
* Classifier and auxiliary heads modified to output two channels (background vs tumor).

#### **Training / pipeline**

* Backbone initially frozen, then fine-tuned.
* Introduced MONAI `DiceLoss` and `DiceMetric`.
* Automatic Mixed Precision (AMP) used.
* Scheduler: `ReduceLROnPlateau`.
* Modular training loops and checkpointing via the project’s `engine.py` (will get way better in V3).

#### **Results**

* Final Validation Accuracy: **93.67%**.
* Best Dice Coefficient: **0.2617**.

#### **Notes / Lessons**

* Transfer learning improved feature extraction and speed of convergence, but **pixel accuracy remained high while Dice stayed low**: confirming the class imbalance problem.
* The Dice implementation and metric handling were corrected in later versions.
* Blurry / poor boundary precision persisted; overfitting started to appear after prolonged training if not controlled.

---

### v2: UNet++ pretrained on EfficientNet-B4 (best archived)

> This model works perfectly, and V3 will build on almost everything here.
> The biggest change will be better organization, modularity and to make a reproducible pipeline with separate folders and files for each component.
#### **Architecture**

* UNet++ decoder with a pretrained EfficientNet-B4 encoder.
* Deep supervision heads (intermediate outputs used for training).
* Advanced loss composition (combined Dice + CE + optional Focal/Tversky variants).

#### **Training / pipeline**

* Augmentations: elastic transforms, grid distortions, CLAHE, noise, flips/rotations.
* Optimizer / scheduler: AdamW + OneCycleLR.
* Mixed Precision (AMP), checkpointing, and rich metrics logging (Dice + Hausdorff).
* Loss / objective: combined losses tailored for class imbalance and boundary accuracy.

#### **Results**

* Final Validation Loss: **0.0974**
* Final Dice Score: **0.7914**
* Final Hausdorff Distance: **21.0548**

> The above metrics come from a **5-epoch run** used to validate the pipeline; they are encouraging but **not** a fully benchmarked result. The architecture typically needs longer training for optimal and stable performance.
#### **Notes / Lessons**

* **Deep supervision** and a stronger **encoder** noticeably improved Dice overlap and boundary performance.
* Carefully designed **augmentations** were key to preventing overfitting.
* **Loss engineering** (Dice/Tversky hybrids, focal variants) had a strong impact on small-region recall and boundary precision.

---

## Compact comparison table

| Version |     Main architecture      | Key changes from previous                                                  |     Reported Dice     | Reported Val Accuracy | Other metric(s)                    |
| ------- | :------------------------: | -------------------------------------------------------------------------- | :-------------------: | :-------------------: | ---------------------------------- |
| v0      |       Vanilla U-Net        | Baseline pipeline and data alignment                                       |     Not reported      |      **90.81%**       | --                                 |
| v1      |   DeepLabV3 (ResNet-50)    | Transfer learning; MONAI Dice; AMP; frozen backbone experiments            |      **0.2617**       |      **93.67%**       | --                                 |
| v2      | UNet++ + (EfficientNet-B4) | Deep supervision, advanced losses, richer augmentations, OneCycleLR, AdamW | **0.7914** (5 epochs) |          --           | Val Loss: 0.0974, Hausdorff: 21.05 |

> **Important:** For segmentation tasks, **Dice** (overlap with ground truth tumor) is the priority metric. Pixel accuracy can be high even for trivial/bad solutions due to class imbalance (tumor's pixels are fewer than background's pixels).

---

## What changed across versions (high level)

1. **Model backbone & capacity**

   * v0: small encoder (U-Net).
   * v1: ResNet-50 backbone (transfer learning).
   * v2: EfficientNet-B4 + UNet++ (stronger pretrained encoder + denser decoder connections).

2. **Loss & metric awareness**

   * v0: BCE: inadequate for small-object segmentation.
   * v1: introduced DiceLoss and Dice metric via MONAI, but results highlighted the need for loss hybrids.
   * v2: combined Dice + CE (+ Focal/Tversky options) to handle imbalance and boundary focus.

3. **Augmentation & regularization**

   * v0: minimal.
   * v1: reproducible transforms and stable training utilities.
   * v2: rich augmentations (elastic/grid/CLAHE), deep supervision, and modern schedulers.

4. **Training utilities**

   * Introduction of AMP, checkpointing and modular engines across v1 → v2; v2 adds OneCycleLR and AdamW for better convergence.


---
## Final notes

* The archived versions show an iterative progression: from a functional baseline (v0), to transfer-learning experiments (v1), to a stronger segmentation pipeline with better loss/augmentations (v2).
* The v2 artifacts are the best preserved and most promising, but remember the reported v2 metrics are from a short run: treat them as **proof of potential**, not final benchmarks.
* Use this README as the source of truth when deciding which components to migrate into `v3` or when reproducing past runs.
