from tqdm import tqdm
import torch
from torch import amp
import monai
from monai.metrics import DiceMetric, HausdorffDistanceMetric


# Training function loop for each epoch
# ---------------------------------------
def train_one_epoch(model,
            train_dataloader,
            loss_fn,
            optim,
            scheduler,
            scaler,
            device):
    # 0. put model in train mode and init total losses
    model.train()
    total_losses = 0
    
    # 1. loop through train_dataloader
    pbar = tqdm(train_dataloader,
                total=len(train_dataloader),
                desc="Training...")

    for batch in pbar:
        
        # 2. move data to device
        imgs = batch['image'].to(device)
        masks = batch['mask'].to(device)

        # 3. enable auto mixed precision (AMP) for efficiency
        with amp.autocast(device_type= device):
            # 4. forard pass
            logits = model (imgs)
            
            # 5. calculate the loss
            loss = loss_fn(logits, masks)
            
            # 6. zero grad
            optim.zero_grad()
            
        # 7. scale loss and back propagate
        scaler.scale(loss).backward()
        
        # 8. step optim, scaler, and scheduler
        scaler.step(optim)
        scaler.update()
        scheduler.step()
        
        # 9. compute total losses
        total_losses += loss.item()
        
        # 10. update pbar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'LR': f'{scheduler.get_last_lr()[0]:.6f}'
        })
        
    # 11. return average loss
    return total_losses/(len(train_dataloader))




# Vaildation function loop
# -------------------------
def validate(model,
            val_dataloader,
            loss_fn,
            metric,
            device):
    
    # 0. put model in evaluation mode
    model.eval()
    
    # 0.0. init losses and reset metrics
    total_losses = 0
    metric['dice'].reset()
    metric['hausdorff'].reset()
    
    # 1. loop through val_dataloader
    with torch.inference_mode():
        pbar = tqdm(val_dataloader,
                    total=len(val_dataloader),
                    desc="Testing...")
        
        for batch in pbar:
            
            # 2. move data to device and reset metrics (start fresh)
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)

            # 3. enable auto mixed precision (AMP) for efficiency
            with amp.autocast(device_type= device):
                # 4. forard pass
                logits = model (imgs)
                
                # 5. calculate the loss
                loss = loss_fn(logits, masks)
            
            # 6. compute total losses
            total_losses += loss.item()
            
            # 7. convert logits -> discrete predictions with consistent shape
            if logits.shape[1] == 1:
                # sigmoid for binary
                probs = torch.sigmoid(logits)
                # binary: keep shape [B, 1, H, W]
                preds = (probs > 0.5).long() 
            else:
                # softmax for multi-class
                probs = torch.softmax(logits, dim=1)
                # multiclass: argmax -> [B, H, W], then add channel dim -> [B, 1, H, W]
                preds = torch.argmax(probs, dim=1).unsqueeze(1).long()
            
            # Skip Empty Masks in Evaluation
            if torch.sum(masks) == 0 and torch.sum(preds) == 0:
                continue
            
            # 8. update metrics directly (MONAI handles one-hot internally)
            metric['dice'](preds, masks)
            metric['hausdorff'](preds, masks)
            
            # 7. update pbar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
            })
            
    # 8. get the final dice_score and hausdorff_distamce
    dice_score = metric['dice'].aggregate()
    hausdorff_vals = metric['hausdorff'].aggregate()

    dice_score = torch.nanmean(dice_score).item()
    hausdorff_dist = torch.nanmean(hausdorff_vals).item()

        
    # 9. return average loss, dice score, and hausdorff distance
    return total_losses/(len(val_dataloader)), dice_score, hausdorff_dist