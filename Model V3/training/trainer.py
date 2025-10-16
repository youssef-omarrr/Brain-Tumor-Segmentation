import torch
from .losses import CombinedLoss
from .training_loop import train_one_epoch, validate
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import os

def train(model,
        train_dataloader,
        val_dataloader,
        load_pretrained:str = None,
        checkpoint_dir:str = "checkpoints/",
        epochs:int = 5,
        max_lr:float = 3e-4):
    
    # 0. init device and move model to it
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # 1. init loss function, optimizer, scheduler, and scaler
    # ---------------------------------------------------------
    # 1.1. loss function
    loss_fn = CombinedLoss()
    
    # 1.2. optimizer
    optim = torch.optim.AdamW(
        model.parameters(),
        lr = max_lr,
        weight_decay= 1e-4,
    )
    
    # 1.3. scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer= optim,
        max_lr= max_lr, # The upper learning rate boundary.
        epochs= epochs,
        steps_per_epoch= len(train_dataloader), # Number of batches in one epoch.
        
        pct_start=0.3,         # Percentage of the cycle spent increasing the LR.
        div_factor=25,         # Determines the initial LR (max_lr / div_factor).
        final_div_factor=100,  # Determines the minimum LR (initial_lr / final_div_factor).
        anneal_strategy='cos'  # Use a cosine annealing strategy for the decay phase.
    )
    
    # 1.4 scaler
    scaler = torch.amp.GradScaler(device=device)
    
    
    # 2. init metric dict for evaluation
    # -----------------------------------
    metric = {
        # 2.1. Dice Metric: Measures the overlap between prediction and ground truth.
        # It is the primary metric for most segmentation tasks.
        'dice': DiceMetric(
            include_background=False, # Crucial: only evaluate the tumor class, not the background.
            reduction='mean',
            get_not_nans=False
        ),
        
        # 2.2. Hausdorff Distance: Measures the distance between the boundaries of the
        # predicted and ground truth segmentations. Excellent for evaluating boundary accuracy.
        'hausdorff': HausdorffDistanceMetric(
            include_background=False,
            reduction='mean',
            percentile=95 # Use the 95th percentile to make the metric robust to outliers.
        )
    }
    
    
    # 3. load pretrained model is available
    # ----------------------------------------
    if load_pretrained:
        if os.path.exists(load_pretrained):
            # load checkpoint
            checkpoint = torch.load(load_pretrained, map_location=device)
            # load model
            model.load_state_dict(checkpoint['model_state_dict'])
            #load optimizer and scheduler
            optim.load_state_dict(checkpoint['optim_state_dict'])
            
            print(f"Loaded pretrained model:")
            print(f"- val_loss={checkpoint['val_loss']:.4f}")
            print(f"- dice_loss={checkpoint['dice_score']:.4f}")
            print(f"- hausdorff_dist={checkpoint['hausdorff_dist']:.4f}")
            
        else:
            print(f"[WARNING] load_pretrained path was provided but does not exist: {load_pretrained}")
            
    # 4. full training loop
    # ----------------------
    for epoch in range(epochs):
        print(f"Training epoch no.{epoch+1} / {epochs}")
        print("-"*35)
        
        # 5. train
        train_loss = train_one_epoch(
            model,
            train_dataloader,
            loss_fn,
            optim,
            scheduler,
            scaler,
            device
        )
        
        # 6. validate
        val_loss, dice_score, hausdorff_dist = validate(
            model,
            val_dataloader,
            loss_fn,
            metric,
            device
        )
        
        # 7. save checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            
            'val_loss': val_loss,
            'dice_score': dice_score,
            'hausdorff_dist': hausdorff_dist,
        }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))

        print("Model saved.")
        
        # 8. print epoch summary
        print(f"Epoch no.{epoch+1} / {epochs} summary")
        print("-"*35)
        print(f"Average train losses = {train_loss:.3f}")
        print(f"Average validation losses = {val_loss:.3f}")
        print(f"Dice Score: {dice_score:.4f}") 
        print(f"Hausdorff Distance: {hausdorff_dist:.4f}") 
        print("="*35, "\n")
        
        