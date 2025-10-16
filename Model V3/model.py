from torch import nn
import segmentation_models_pytorch as smp

# Model defination
# -----------------
class TumorSegmentaionModel(nn.Module):
    """
    A segmentation model using a UNet++ decoder with a pretrained EfficientNet-b4 encoder.
    Implements deep supervision for improved training performance.
    
    UNet++ improves over U-Net by introducing nested skip connections, which help the model learn finer details.
    
    Deep supervision adds auxiliary loss branches at intermediate decoder layers, making training more stable and faster to converge.
    """
    def __init__(self, 
                num_classes:int = 2,
                encoder:str = "efficientnet-b4",
                deep_supervision:bool = True):
        super().__init__()
        
        # 1. init the main model from segmentation_models_pytorch library
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder,           # The backbone network.
            encoder_weights="imagenet",     # Use pretrained weights for transfer learning.
            in_channels=1,                  # Grayscale.
            classes=num_classes,            # out is only 2 -> tumor, background
            activation=None,                # will be handled by loss function for better generality
        )
        
        # 2. Get the number of output channels from each stage of the encoder.
        """
        This makes the model adaptable to different encoder backbones.
        
        e.g. for "efficientnet-b4"
        encoder_channels -> [1 -> (input channels 'grayscale'), 
                            24 -> (o/p after first block), 
                            32, 56, 160, 448 -> (progressively deeper layers with more channels)
                            ]
        """
        encoder_channels = self.model.encoder.out_channels 
        
        # 3. define auxiliary heads for deep supervision.
        self.deep_supervision = deep_supervision
        if deep_supervision:
            # Each auxiliary head is a 1x1 convolution that changes
            # the number of channels in that feature map so it matches the number of classes (2: tumor, background)
            self.aux_head1 = nn.Conv2d(
                encoder_channels[-2],
                num_classes,
                kernel_size=1
            )
            self.aux_head2 = nn.Conv2d(
                encoder_channels[-3],
                num_classes,
                kernel_size=1
            )
            
    def forward(self, x):
        # 1. main forward pass
        # ---------------------
        main_output = self.model(x)
        
        # 2. deep supervision (for training only)
        if self.training and self.deep_supervision:
            
            # 2.1. Get the feature maps from different stages of the encoder.
            """
            features = [
                (B, 3, 512, 512),      # input
                (B, 24, 256, 256),     # shallow
                (B, 32, 128, 128),     # deeper
                (B, 56, 64, 64),       # features[-3]
                (B, 160, 32, 32),      # features[-2]
                (B, 448, 16, 16)       # deepest
            ]
            """
            features = self.model.encoder(x)
            
            # 2.2. Create auxiliary predictions from intermediate feature maps.
            # These features correspond to different spatial scales in the network.
            """
            aux1 -> (B, num_classes, 32, 32)
            aux2 -> (B, num_classes, 64, 64)
            """
            aux1 = self.aux_head1(features[-2]) # Feature map at 1/16 resolution
            aux2 = self.aux_head2(features[-3]) # Feature map at 1/8 resolution
            
            # 2.3. Upsample these auxiliary predictions to match the main output's spatial dimension
            aux1 = nn.functional.interpolate(
                            aux1,
                            size=main_output.shape[2:], # (B, num_classes, 32, 32) change the last 2 dimensions only
                            mode="bilinear", # a smooth way of resizing 2D images.
                            align_corners=False,
            )
            aux2 = nn.functional.interpolate(
                            aux2,
                            size=main_output.shape[2:], # (B, num_classes, 64, 64) change the last 2 dimensions only
                            mode="bilinear",
                            align_corners=False,
            )
            
            # 2.4. return dict with all outputs (now that they have the same dimensions)
            return {
                'main': main_output,
                'aux1': aux1,
                'aux2': aux2,
            }
            
        # 3. during validation/inference: only return the main output
        return main_output