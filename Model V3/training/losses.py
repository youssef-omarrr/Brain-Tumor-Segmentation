from torch import nn
from monai.losses import DiceCELoss, FocalLoss, TverskyLoss

"""
1. `DiceCELoss`: Combines **Dice Loss** (handles class imbalance between tumor and background) 
    and **Cross-Entropy Loss** (focuses on pixel accuracy). 
    This is the main loss used for training.

2. `FocalLoss`: Makes the model focus more on **hard pixels** (like tumor edges) 
    by reducing the loss from easy pixels.

3. `TverskyLoss`: A version of Dice Loss that lets us control the balance between 
    **false positives** and **false negatives**. 
    Here, it’s set to penalize false positives more (to avoid marking healthy tissue as tumor).
"""
class CombinedLoss(nn.Module):
    def __init__(self, 
                dice_ce_weight:float = 0.5,
                focal_weight:float = 0.3,
                tversky_weight:float = 0.2,
                ):
        super().__init__()

        # 1. define weights for combining loss components
        self.alpha = dice_ce_weight     # Weight for Dice+CE loss
        self.beta = focal_weight        # Weight for Focal loss
        self.gamma = tversky_weight     # Weight for Tversky loss

        # 2. init individual loss functions 
        # ----------------------------------
        
        # 2.1. Dice + CrossEntropy
        self.dice_ce = DiceCELoss(
            include_background=False,   # Exclude background class from Dice calculation (since most pixels are background).
            to_onehot_y=True,           # Convert target to one-hot format (needed for multi-class training).
            softmax=True,               # Apply softmax to predictions.
            lambda_dice=0.5,            # Weight for Dice component.
            lambda_ce=0.5,              # Weight for CrossEntropy component.
            squared_pred=True,          # Use squared predictions in Dice denominator for smoother gradients.
            smooth_dr=1e-5,             # Small constants to avoid division by zero.
            smooth_nr=1e-5,             # Small constants to avoid division by zero.
        )
        
        # 2.2. Focal Loss
        self.focal = FocalLoss(
            include_background=False,
            to_onehot_y=True,
            alpha=0.75,         # Balances importance between classes (helps with imbalance)
            gamma=2.0,          # Controls how much to focus on hard examples (higher = more focus).
            reduction='mean',   # Average the loss over the batch.
        )
        
        # 2.3. Tversky Loss
        self.tversky = TverskyLoss(
            include_background=False,
            to_onehot_y=True,
            alpha=0.7,  # Penalizes false positives more heavily.
            beta=0.3,   # Penalizes false negatives less heavily.
            smooth_dr=1e-5,
            smooth_nr=1e-5,
        )
        
    # Helper compute loss function
    # ------------------------------
    def _compute_loss(self, pred, target):
        """
        Helper method to calculate the weighted sum of the three loss components.
        Combine the individual losses using the predefined alpha, beta, and gamma weights.
        """
        dice_ce_loss = self.dice_ce(pred, target)
        focal_loss = self.focal(pred, target)
        tversky_loss = self.tversky(pred, target)
        
        # return total loss as ensamble
        return (
            self.alpha * dice_ce_loss +
            self.beta  * focal_loss   + 
            self.gamma * tversky_loss
            )
        
        
    # forward function
    # -----------------
    def forward(self, pred, target):
        # 1. handling deep supervision (in training)
        if isinstance(pred, dict): # if the prediction is a dict.
            main_loss = self._compute_loss(pred['main'], target)
            aux1_loss = self._compute_loss(pred['aux1'], target)
            aux2_loss = self._compute_loss(pred['aux2'], target)
            
            # combine the losses of the 3 outputs, with less weight for the aux outputs (change the weights if needed)
            total_loss = main_loss + (0.4 * aux1_loss) + (0.2 * aux2_loss)     
            return total_loss    
        
        # 2. handle standard output (infernce)
        else:
            return self._compute_loss(pred, target)