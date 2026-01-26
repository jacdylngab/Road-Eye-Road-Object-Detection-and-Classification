import torch.nn as nn
from typing import Sequence, Tuple

INF = 1e8

class FCOSHead(nn.Module):
    """
    Implementing an Anchor Free Head. This is based on `"FCOS: Fully Convolutional One-Stage Object Detection" <https://arxiv.org/abs/1904.01355>`_.
    The code is also based on `https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/fcos_head.py`

    ADD MORE STUFF HERE LATER!
    """
    def __init__(self,
                 num_clases: int,
                 in_channels: int,
                 feat_channels: int = 256,
                 stacked_convs: int = 4,
                 strides: Sequence[int] = (4, 8, 16, 32, 64),
                 regression_ranges: Sequence[Tuple[int, int]] = ((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)),
                 center_sampling: bool = True, # Points near the center are positive others are not if this is true
                 center_sample_radius: float = 1.5, 
                 norm_on_bbox: bool = True, # Normalize targets (l,t,b,r) by stride
                 centerness_on_reg: bool = True, # Centerness is predicted on box regression not classification
                 dcn_on_last_conv: bool = False, # Deformable Convolution Network (dcn) allows the kernel to shift its sampling locations dynamically, which is useful for detecting objects of different shapes or irregular geometry.
                 conv_bias: Union[bool, str] = 'auto', # Bias in a convolution is just a learned scalar added after the weighted sum. 'auto' means the code will decide automatically based on whether you have normalization (like BatchNorm or GroupNorm) right after the conv.
                 loss_cls: dict = ( # Classification Loss that answers what the object is
                     type='FocalLoss', # Downweight easy examples (background) and focus on hard ones
                     use_sigmoid=True, # used for multi-label classification
                     gamma=2.0, # How hard we downweight easy examples
                     alpha=0.25, # Balance positives vs negatives
                     loss_weight=1.0 # It is a multiplier. Controls how much the loss contributes to the final loss
                     ),
                loss_bbox: dict = ( # Regression loss that answers where the object is
                    type='IoULoss', # How much do the predicted box and GT box overlap
                    loss_weight=1.0),
                loss_centerness: dict = ( # Centerness loss to find the best center on the object
                    type="CrossEntropy", # Center points produce better bounding. While edge points produce bad boxes
                    use_sigmoid=True,
                    loss_weight=1.0),
                bbox_coder: dict = (type='DistancePointBBoxCoder'), # it “decodes” the predicted offsets into (x1, y1, x2, y2) coordinates and can also “encode” ground truth boxes into a format the model can learn from. 
                normalization_cfg: dict = ( # This defines how normalization layers in the head are configured
                    type='GN', # Group Normalization. It divides channels into groups and normalizes them separately. 
                    num_groups=32, # How many group to divide channels into 
                    requires_grad=True # The scale (γ) and shift (β) parameters in GN are learnable (the network can adjust them).
                    ),
                init_cfg: dict = ( # This defines how to initialize weights in the head.
                    type='Normal', # Use normal (Gaussian) distribution to initialize weights
                    name='Conv2d', # Applies to all conv layers
                    std=0.01, # Standard deviation of the normal distribution. Small numbers -> start with small weights
                    override = dict( # Special rule for classification conv (conv_cls)
                        type='Normal',
                        name='conv_cls',
                        std=0.01,
                        bias_prob=0.01 # Initialize biases so that probability of predicting foreground class is very low initially
                        )),
                ) -> None:
        """
        Docstring for __init__
        
        :param self: Description
        """
        pass
