# ====================================================================================================== #
# IMPORTS
# ====================================================================================================== #

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from pathlib import Path
import cv2
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
from monai.losses import DiceCELoss, FocalLoss, TverskyLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# ====================================================================================================== #
# TRANSFORMS
# ====================================================================================================== #

def get_train_transforms():
    """
    Defines the augmentation pipeline for the training dataset.
    Uses a rich set of transforms to improve model robustness and generalization.
    """
    return A.Compose([
        # --- 1. Geometric Transformations ---
        # These alter the spatial orientation of the image.
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT, # Fill new pixels with a constant value
            value=0,
            mask_value=0,
            p=0.5
        ),

        # --- 2. Non-rigid (Elastic) Transformations ---
        # These simulate tissue deformation, which is common in medical scans.
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.3
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.3
        ),

        # --- 3. Intensity and Color Transformations ---
        # These alter pixel values to simulate different lighting/scanner conditions.
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # --- 4. Noise and Blurring ---
        # Simulates sensor noise and minor focus issues.
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),

        # --- 5. Advanced Contrast Enhancement ---
        # CLAHE is highly effective for enhancing local contrast in medical images.
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

        # --- 6. Final Preprocessing Steps ---
        # Normalize the image using ImageNet stats (a common starting point).
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        # Convert the image and mask to PyTorch Tensors.
        ToTensorV2(),
    ])


def get_val_transforms():
    """
    Defines the transformation pipeline for the validation dataset.
    Only includes essential preprocessing steps, no random augmentations.
    """
    return A.Compose([
        # 1. Normalize the image to match the training data distribution.
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        # 2. Convert the image and mask to PyTorch Tensors.
        ToTensorV2(),
    ])

# ====================================================================================================== #
# MODEL ARCH
# ====================================================================================================== #

class BrainTumorSegmentationModel(nn.Module):
    """
    A segmentation model using a UNet++ decoder with a pretrained EfficientNet encoder.
    Implements deep supervision for improved training performance.
    """
    def __init__(self, num_classes=2, encoder_name="efficientnet-b4"):
        super().__init__()
        
        # 1. Initialize the main model from the segmentation-models-pytorch (smp) library.
        # UNet++ often provides better performance than standard U-Net due to its nested skip pathways.
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,      # The backbone network (e.g., 'efficientnet-b4').
            encoder_weights="imagenet",     # Use pretrained weights for transfer learning.
            in_channels=3,                  # Expects 3-channel (RGB) input images.
            classes=num_classes,            # Number of output classes (e.g., 2 for background/tumor).
            activation=None,                # Output raw logits; activation is handled by the loss function.
        )
        
        # Get the number of output channels from each stage of the encoder.
        # This makes the model adaptable to different encoder backbones.
        encoder_channels = self.model.encoder.out_channels
        
        # 2. Define auxiliary heads for deep supervision.
        # This technique adds loss signals at intermediate decoder layers to improve gradient flow.
        self.deep_supervision = True
        if self.deep_supervision:
            # These heads are simple 1x1 convolutions that map intermediate feature maps
            # from the decoder to the desired number of output classes.
            self.aux_head1 = nn.Conv2d(encoder_channels[-2], num_classes, kernel_size=1)
            self.aux_head2 = nn.Conv2d(encoder_channels[-3], num_classes, kernel_size=1)
    
    def forward(self, x):
        # --- Main Forward Pass ---
        main_output = self.model(x)
        
        # --- Deep Supervision (only during training) ---
        if self.training and self.deep_supervision:
            # Get the feature maps from different stages of the encoder.
            # Note: This is a simplified implementation; it re-runs the encoder.
            features = self.model.encoder(x)
            
            # Create auxiliary predictions from intermediate feature maps.
            # These features correspond to different spatial scales in the network.
            aux1 = self.aux_head1(features[-2])  # Feature map at 1/16 resolution
            aux2 = self.aux_head2(features[-3])  # Feature map at 1/8 resolution
            
            # Upsample the auxiliary predictions to match the main output's spatial dimensions.
            aux1 = F.interpolate(aux1, size=main_output.shape[2:], mode='bilinear', align_corners=False)
            aux2 = F.interpolate(aux2, size=main_output.shape[2:], mode='bilinear', align_corners=False)
            
            # Return a dictionary of outputs. The loss function will handle combining them.
            return {
                'main': main_output,
                'aux1': aux1,
                'aux2': aux2
            }
        
        # During validation or inference, only return the final, main prediction.
        return main_output

# ====================================================================================================== #
# PRED AND PLOT
# ====================================================================================================== #

def predict_and_visualize(model, image_path, mask_path=None, device='cuda', threshold=0.5):
    """
    Make prediction and visualize results
    """
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    
    # Apply same normalization as validation
    transform = get_val_transforms()
    transformed = transform(image=image, mask=np.zeros((512, 512)))
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        with torch.amp.autocast(device_type= device):
            output = model(input_tensor)
            if isinstance(output, dict):
                output = output['main']
            
            pred_probs = torch.softmax(output, dim=1)
            pred_mask = (pred_probs[0, 1] > threshold).cpu().numpy()
    
    # Visualization
    fig, axes = plt.subplots(1, 3 if mask_path else 2, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(pred_mask, cmap='jet', alpha=0.7)
    axes[1].imshow(image, alpha=0.3)
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    if mask_path:
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        true_mask = cv2.resize(true_mask, (512, 512))
        true_mask = (true_mask > 127).astype(np.uint8)
        
        axes[2].imshow(true_mask, cmap='jet', alpha=0.7)
        axes[2].imshow(image, alpha=0.3)
        axes[2].set_title('Ground Truth Mask')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return pred_mask

# ====================================================================================================== #
# MAIN
# ====================================================================================================== #

model = BrainTumorSegmentationModel()

checkpoint = torch.load("checkpoints/best_modelV2.pth", map_location="cuda")
model.load_state_dict(checkpoint['model_state_dict'])
model.to("cuda")

import random
# Pick random test image
pic_number = random.randint(1, 3064)
IMAGE_PATH = f"brain_tumor_dataset/images/{pic_number}.png"
MASK_PATH = f"brain_tumor_dataset/masks/{pic_number}.png"

predict_and_visualize(
    model= model,
    image_path= IMAGE_PATH,
    mask_path= MASK_PATH
)