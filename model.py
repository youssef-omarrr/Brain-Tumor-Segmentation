import torch
import matplotlib.pyplot as plt
from monai.transforms import (
    LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, Compose
)
import numpy as np
import random
pic_number = random.randint(1, 3064)

# ==== CONFIG ====
MODEL_PATH = "checkpoints/model_epoch_20.pth"
IMAGE_PATH = f"brain_tumor_dataset/images/{pic_number}.png" 
MASK_PATH = f"brain_tumor_dataset/masks/{pic_number}.png" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = (256, 256)  # same as training

# ==== MODEL (example UNet – use your own if different) ====
from monai.networks.nets import UNet
model = UNet(
    spatial_dims=2,                 # Using 2D convolutions and operations.
    
    in_channels=1,        # Number of channels in the input image.
    out_channels=1,      # Number of channels in the output mask.
    
    channels=(16, 32, 64, 128, 256),# Feature maps at each level of the encoder.
    strides=(2, 2, 2, 2),           # Downsampling factor at each encoder level.
    
    num_res_units=2,                # Number of convolutional blocks per level.
).to(DEVICE) # Move the model to the specified device (GPU or CPU).

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==== TRANSFORMS ====
preprocess = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize(spatial_size=TARGET_SIZE),
])

# Load once to get original size (for back-resize)
orig_img = LoadImage(image_only=True)(IMAGE_PATH)
orig_mask = LoadImage(image_only=True)(MASK_PATH)
orig_h, orig_w = orig_img.shape[:2]

# ==== PREP INPUT ====
img_t = preprocess(IMAGE_PATH)                  # C,H,W (float32 in [0,1] after scaling)
img_batch = torch.as_tensor(img_t).unsqueeze(0).to(DEVICE)  # 1,C,H,W

# ==== INFERENCE ====
with torch.no_grad():
    logits = model(img_batch)                   # 1,1,h,w
    prob = torch.sigmoid(logits)[0, 0].cpu()    # h,w
bin_mask = (prob > 0.5).float()                 # h,w

# ==== VISUALIZATION ====
# Resize predicted mask back to original image size for correct overlay
resize_back = Resize(spatial_size=(orig_h, orig_w), mode="nearest")
# bin_mask is a (H, W) tensor, needs to be (C, H, W) for Resize
bin_mask_resized = resize_back(bin_mask.unsqueeze(0)).squeeze(0).numpy()

# Binarize the original mask for consistent display
orig_mask_binary = (orig_mask > 127).astype(np.uint8)

# Create plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(orig_img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(bin_mask_resized, cmap='jet', alpha=0.7)
axes[1].imshow(orig_img, cmap='gray', alpha=0.3)
axes[1].set_title('Predicted Mask')
axes[1].axis('off')

axes[2].imshow(orig_mask_binary, cmap='jet', alpha=0.7)
axes[2].imshow(orig_img, cmap='gray', alpha=0.3)
axes[2].set_title('Ground Truth Mask')
axes[2].axis('off')

plt.tight_layout()
plt.show()
