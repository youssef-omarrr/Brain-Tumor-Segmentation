from .datasets import *
from numpy import random
from torch.utils.data import Subset, DataLoader

def create_dataloaders(
                    imgs_dir:str,
                    masks_dir:str,
                    batch_size:int = 4,
                    train_split:int = 0.8,
                    img_size:tuple = (512, 512),
                    random_seed:int = 42):
    """
    Creates train and validation DataLoaders from MRI images and masks.

    Args:
        imgs_dir (str): Path to images.
        masks_dir (str): Path to masks.
        batch_size (int): Batch size. Default is 8.
        train_split (float): Train/val split ratio. Default is 0.8.
        img_size (tuple): Resize shape. Default is (512, 512).
        random_seed (int): Seed for reproducibility. Default is 42.

    Returns:
        (DataLoader, DataLoader): Train and validation loaders.
    """
    
    # 1. get 2 datasets one for training and the other for validation
    # This still scans the directory twice but is necessary to assign different transforms.
    train_dataset = TumorDataset(
        imgs_dir=imgs_dir,
        masks_dir=masks_dir,
        
        transform=train_transforms(),
        img_size=img_size
    )
    
    val_dataset = TumorDataset( # using the default transforms
        imgs_dir=imgs_dir,
        masks_dir=masks_dir,
        
        img_size=img_size
    )
    
    # 2. create a random split of indices
    dataset_size = len(train_dataset) # total dataset size
    indices = list(range(dataset_size)) # list of all indices [0, 1, 2, ..., dataset_size - 1]
    
    split = int(train_split * dataset_size)
    random.seed(random_seed)
    random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    
    # 3. create pytorch subsets with the split indices
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # 4. create dataloades from subsets
    train_dataloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    
    val_dataloader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle= False,
        pin_memory=True,
    )
    
    print(f"Train dataset size: {len(train_subset)}")
    print(f"Validation dataset size: {len(val_subset)}")
    
    return train_dataloader, val_dataloader