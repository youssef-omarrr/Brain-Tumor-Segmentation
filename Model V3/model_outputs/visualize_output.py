import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_val_transforms_gray():
    """
    Validation transforms for grayscale images.
    Normalizes to mean=0.5, std=0.5 and converts to tensor.
    """
    return A.Compose([
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ])


def visualize_random_prediction(model, images_dir, masks_dir=None, device="cuda", threshold=0.5):
    """
    Pick a random grayscale image, predict its mask, and visualize the result.

    Args:
        model: Trained segmentation model.
        images_dir: Directory containing input images (.png).
        masks_dir: Directory containing corresponding masks (optional).
        device: "cuda" or "cpu".
        threshold: Probability cutoff for mask.
    """
    model.eval().to(device)

    # Pick random image
    image_files = sorted(list(Path(images_dir).glob("*.png")))
    if not image_files:
        raise ValueError(f"No images found in {images_dir}")
    img_path = random.choice(image_files)

    mask_path = Path(masks_dir) / img_path.name if masks_dir else None

    # Load grayscale image
    image = Image.open(img_path).convert("L")  # L = grayscale
    image = image.resize((512, 512))
    image_np = np.array(image)

    # Albumentations expects (H, W, C)
    image_np = np.expand_dims(image_np, axis=-1)

    transform = get_val_transforms_gray()
    transformed = transform(image=image_np, mask=np.zeros((512, 512)))
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    # Predict
    with torch.inference_mode():
        output = model(input_tensor)
        if isinstance(output, dict):
            output = output['main']
        probs = torch.softmax(output, dim=1)
        pred_mask = (probs[0, 1] > threshold).cpu().numpy()

    # Plot results
    n_cols = 3 if mask_path and mask_path.exists() else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(15, 6))

    # Original image
    axes[0].imshow(image_np.squeeze(), cmap="gray")
    axes[0].set_title("Original Image", fontsize=15)
    axes[0].axis("off")

    # Predicted mask
    axes[1].imshow(image_np.squeeze(), cmap="gray", alpha=0.5)
    axes[1].imshow(pred_mask, cmap="jet", alpha=0.5)
    axes[1].set_title("Predicted Mask", fontsize=15)
    axes[1].axis("off")

    # Ground truth mask
    if mask_path and mask_path.exists():
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((512, 512))
        mask_np = (np.array(mask) > 127).astype(np.uint8)

        axes[2].imshow(image_np.squeeze(), cmap="gray", alpha=0.5)
        axes[2].imshow(mask_np, cmap="jet", alpha=0.5)
        axes[2].set_title("Ground Truth Mask", fontsize=15)
        axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Predicted image: {img_path.name}")
    return pred_mask
