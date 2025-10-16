# Albumentations Augmentation Notes

A detailed reference for the most common augmentations used in segmentation and medical image preprocessing pipelines.

---

## 1. **Geometric Transformations**

These transformations modify the **spatial structure** (position, orientation, scale) of the image and mask.

### `A.RandomRotate90(p=0.5)`

* **Description:** Rotates the image by 90°, 180°, or 270° randomly.
* **Parameters:**

  * `p`: Probability of applying the transform.
* **Use Cases:**

  * Works well for **square images** (e.g., MRI slices).
  * Introduces **rotation invariance** for classification and segmentation models.

---

### `A.HorizontalFlip(p=0.5)`

* **Description:** Randomly flips the image horizontally.
* **Parameters:**

  * `p`: Probability of flipping.
* **Use Cases:**

  * Common for natural and medical images.
  * Helps when left/right orientation isn’t important (e.g., brain MRIs).

---

### `A.VerticalFlip(p=0.5)`

* **Description:** Randomly flips the image vertically.
* **Parameters:**

  * `p`: Probability of flipping.
* **Use Cases:**

  * For **top-bottom invariant** images like microscopy or organ scans.

---

### `A.Transpose(p=0.5)`

* **Description:** Swaps image axes, equivalent to rotating by 90° + transposing.
* **Parameters:**

  * `p`: Probability of transposing.
* **Use Cases:**

  * Adds variety to datasets where images are not orientation-specific.

---

### `A.ShiftScaleRotate(...)`

Performs **translation**, **scaling**, and **rotation** in a single operation.

**Parameters:**

* `shift_limit`: Fraction of image size to shift along x/y.
  *(e.g., `0.0625` means up to 6.25% shift)*
* `scale_limit`: Range for zooming in/out (e.g., ±10%).
* `rotate_limit`: Max rotation in degrees (±15° here).
* `border_mode`: How to fill empty pixels after rotation/shift.

  * `cv2.BORDER_CONSTANT` (default) fills with a constant color.
* `value`: Fill color for **image** (e.g., `0` for black background).
* `mask_value`: Fill value for **mask** (typically `0` for background).
* `p`: Probability of applying transform.

**Use Cases:**

* Widely used in **segmentation and classification**.
* Improves robustness to small rotations, scaling, or translations.
* Medical: simulates small patient or sensor movement.

---

## 2. **Non-Rigid (Elastic) Transformations**

Used to simulate **realistic deformations**, e.g., organ movement or tissue distortion.

### `A.ElasticTransform(...)`

Applies smooth elastic deformation to the image and mask.

**Parameters:**

* `alpha`: Magnitude of displacement (controls *strength* of warping).
* `sigma`: Gaussian smoothing factor (controls *smoothness* of warping).
* `alpha_affine`: Strength of additional global affine transform before warping.
* `border_mode`: How to fill pixels outside image borders.
* `value`: Fill color for image.
* `mask_value`: Fill color for mask.
* `p`: Probability.

**Use Cases:**

* Simulates **biological tissue deformation**.
* Great for **MRI, CT**, and **ultrasound** images.
* Adds realism for segmentation tasks (shape variability).

---

### `A.GridDistortion(...)`

Splits image into a grid and distorts each cell slightly.

**Parameters:**

* `num_steps`: Grid size (e.g., 5×5 grid).
* `distort_limit`: Max displacement factor for grid cells.
* `border_mode`: Fill mode for borders.
* `value`: Fill color for image.
* `mask_value`: Fill value for mask.
* `p`: Probability.

**Use Cases:**

* Simulates **lens distortion**, **nonlinear stretching**, or **scanner warping**.
* Useful in **medical**, **OCR**, and **aerial** datasets.

---

## 3. **Intensity and Color Transformations**

Affect pixel intensity or color values, improves robustness to lighting, scanner calibration, or exposure variation.

### `A.RandomBrightnessContrast(...)`

Randomly modifies both brightness and contrast.

**Parameters:**

* `brightness_limit`: Range for brightness adjustment (e.g., ±0.2 → ±20%).
* `contrast_limit`: Range for contrast adjustment.
* `p`: Probability.

**Use Cases:**

* Handles **lighting differences** between samples.
* Common for **X-ray**, **histology**, or **microscopy** datasets.

---

### `A.RandomGamma(...)`

Applies gamma correction with a random gamma value.

**Parameters:**

* `gamma_limit`: Range for gamma (e.g., `(80,120)` means γ∈[0.8,1.2]).
* `p`: Probability.

**Use Cases:**

* Adjusts **image brightness non-linearly**.
* Corrects underexposed or overexposed images.
* Enhances visibility of **soft-tissue** in medical images.

---

## 4. **Noise and Blurring**

Simulate **sensor noise**, **motion blur**, and **imperfect focus** conditions.

### `A.GaussNoise(...)`

Adds random Gaussian noise to the image.

**Parameters:**

* `var_limit`: Variance range of noise (controls intensity).
* `p`: Probability.

**Use Cases:**

* Simulates **low-quality imaging sensors**.
* Prevents model overfitting to clean data.
* MRI, ultrasound, low-dose CT applications.

---

### `A.GaussianBlur(...)`

Applies Gaussian blur (low-pass filter).

**Parameters:**

* `blur_limit`: Max kernel size (odd integer).
* `p`: Probability.

**Use Cases:**

* Simulates **motion blur** or **focus loss**.
* Used to increase robustness to out-of-focus images.

---

## 5. **Advanced Contrast Enhancement**

### `A.CLAHE(...)`

(Contrast Limited Adaptive Histogram Equalization)

CLAHE is a powerful **local contrast enhancement** technique that improves visibility in regions with *poor lighting* or *low dynamic range*.

Unlike global histogram equalization (which enhances the entire image uniformly), CLAHE works on **small tiles** (subregions), enhancing contrast locally and then blending the tiles together.

**How it Works:**

1. Divide the image into small rectangular regions (tiles) defined by `tile_grid_size`. For example, `(8,8)` divides the image into 8×8 tiles.

2. Perform **histogram equalization** independently on **each tile**, spreading out the intensity values within that tile.

3. Clip the histogram at `clip_limit` to avoid over-amplifying noise in uniform areas (like smooth tissue in an MRI).
   
4. Interpolate between tiles to remove blocky artifacts and create a seamless transition across boundaries.

**Parameters:**

* `clip_limit`: Threshold for contrast limiting (higher → stronger contrast).
* `tile_grid_size`: Size of grid for histogram equalization (e.g., `(8,8)`).
* `p`: Probability.

**Use Cases:**

* Excellent for **medical imaging** (MRI, X-ray, CT).
* Reveals **local details** in low-contrast or unevenly illuminated images.

---

##  Summary Table

| Category  | Transform                                     | Main Goal                  | Key Params                                   | Typical Use Case            |
| --------- | --------------------------------------------- | -------------------------- | -------------------------------------------- | --------------------------- |
| Geometric | `RandomRotate90`                              | Rotation invariance        | `p`                                          | Classification/Segmentation |
| Geometric | `HorizontalFlip`, `VerticalFlip`, `Transpose` | Orientation invariance     | `p`                                          | General, Medical Imaging    |
| Geometric | `ShiftScaleRotate`                            | Position, scale, rotation  | `shift_limit`, `scale_limit`, `rotate_limit` | Spatial robustness          |
| Elastic   | `ElasticTransform`                            | Nonlinear deformation      | `alpha`, `sigma`, `alpha_affine`             | MRI, tissue warping         |
| Elastic   | `GridDistortion`                              | Grid-based deformation     | `num_steps`, `distort_limit`                 | Scanner distortion          |
| Intensity | `RandomBrightnessContrast`                    | Lighting variation         | `brightness_limit`, `contrast_limit`         | Illumination correction     |
| Intensity | `RandomGamma`                                 | Exposure correction        | `gamma_limit`                                | Low-light scans             |
| Noise     | `GaussNoise`                                  | Sensor noise simulation    | `var_limit`                                  | MRI/CT realism              |
| Blur      | `GaussianBlur`                                | Motion or focus blur       | `blur_limit`                                 | Real-world robustness       |
| Contrast  | `CLAHE`                                       | Local contrast enhancement | `clip_limit`, `tile_grid_size`               | Medical imaging enhancement |

---

