from torch.utils.data import Dataset
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
import numpy as np


# Dataset Class
# --------------
class TumorDataset(Dataset):
    """
    Custom PyTorch Dataset for loading paired (image, mask) data used in tumor detection (e.g., brain MRI).
    Each image must have a corresponding mask file with the same filename for proper pairing.
    """
    
    # init function
    # ---------------
    def __init__(self,
                imgs_dir:str,
                masks_dir:str,
                transform = None,
                img_size: tuple= (512, 512)):
        super().__init__()
        
        # 1. init paths and params
        self.img_dir = Path(imgs_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        
        # 2.1. init default transform (concats the custom transform with the default one)
        default_transform = [
            # resize image 
            A.Resize(height= self.img_size[0], 
                    width=self.img_size[1]),
            
            # normalize images to grayscale normalization
            A.Normalize(mean=[0.5],
                        std=[0.5],
                        max_pixel_value=255.0),
            
            # convert to pytorch tensor
            A.ToTensorV2(),
            
        ]
        
        # 2.2. concat with user's transforms (if given)
        if transform is not None:
            self.transform = A.Compose([
                *transform, # * -> gets items inside the list instead of the list itself
                *default_transform, 
            ])
        else:
            self.transform = A.Compose(default_transform)
            
        
        # 3. get all the images in the img_dir 
        self.img_files = sorted ( list(self.img_dir.glob("*.png")) )
        
        # 4. make sure every img file has a corresponding mask file
        self.valid_files = []
        
        for img_path in self.img_files:
            mask_path = self.masks_dir/img_path.name # *the mask and image should have the same name for this to work
            
            if mask_path.exists():
                self.valid_files.append(img_path)
                
        # print (f"Found {len(self.valid_files)}/{len(self.img_files)} vaild img-mask pairs.")
        
    
    # get len function
    # ------------------
    def __len__(self):
        # return the ttal number of vaild pairs
        return len(self.valid_files)
    
    # get item function
    # -------------------
    def __getitem__(self, idx):
        # 1. Load paths 
        img_path = self.valid_files[idx]
        mask_path = self.masks_dir / img_path.name
        
        # 2. Load imgs as grayscale (1 - channel)
        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        # 3. Convert to numpy (for albumentation to work)
        img = np.array(img)
        mask = np.array(mask)
        
        # 4. Binzarize mask (1, 0 only)
        mask = (mask > 127).astype(np.uint8)
        
        # 5. Apply albumentation transforms (expects dict, returns dict)
        augmented = self.transform(image = img,
                                mask = mask)
        
        img = augmented["image"]
        mask = augmented["mask"]
        
        # 6. Ensure mask's diemnsion is correct [1, H, W] and dtype long (to avoid errors in the loss function)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask = mask.long()
        
        # 7. return item as dict
        return{
            "image": img,   # torch.FloatTensor [1, H, W]
            "mask": mask,   # torch.LongTensor [1, H, W]
            "file_name": img_path.name,
        }
        
        

# Data Augmentation
# ------------------
from cv2 import BORDER_CONSTANT
def train_transforms():
    """
    Defines data augmentation for the training dataset to improve model generalization.
    The validation dataset uses only the default transformations.

    Most parameters use default values but are explicitly written for clarity and practice.
    """
    
    return [
        # 1. geometric trans
        # --------------------
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.Affine( # new version of 'ShiftScaleRotate'
            translate_percent=0.0625,   # same as shift_limit
            scale=(0.9, 1.1),           # equivalent to scale_limit=0.1
            rotate=(-15, 15),           # same as rotate_limit
            fit_output=False,           # same as keeping image size fixed
            p=0.5
        ),

        
        # 2. non-rigid (elastic) trans
        # ------------------------------
        A.ElasticTransform(
            alpha= 1,
            sigma=50,
            border_mode= BORDER_CONSTANT,
            p = 0.3 
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            border_mode= BORDER_CONSTANT,
            p = 0.3 
        ),
        
        # 3. intensity and color trans
        # ------------------------------
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.RandomGamma(
            gamma_limit=(80, 120),
            p=0.3
        ),
        
        # 4. noise and blurring
        # ----------------------
        A.GaussNoise(
            std_range=(0.05, 0.1),  # equivalent strength; range as a fraction of max value
            mean_range=(0.0, 0.0),  # keep mean centered
            p=0.3
        ),
        A.GaussianBlur(
            blur_limit=3,
            p=0.2
        ),
        
        # 5. Advanced contrast enhancement 
        # ----------------------------------
        # CLAHE is highly effective for enhancing local contrast in medical images.
        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size= (8,8),
            p=0.3
        ),
        
    ]