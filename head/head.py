import torch.nn as nn
from typing import Optional, Sequence, Tuple, Dict, Any

INF = 1e8

class FCOSHead(nn.Module):
    """
    Implementing an Anchor Free Head. This is based on `"FCOS: Fully Convolutional One-Stage Object Detection" <https://arxiv.org/abs/1904.01355>`_.
    The code is also based on `https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/fcos_head.py`

    ADD MORE STUFF HERE LATER!
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 feat_channels: int = 256,
                 stacked_convs: int = 4,
                 strides: Sequence[int] = (4, 8, 16, 32, 64),
                 regression_ranges: Sequence[Tuple[int, int | float]] = ((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)),
                 center_sampling: bool = True, # Points near the center are positive in the GT others are not if this is true. Helps in training
                 center_sample_radius: float = 1.5, 
                 norm_on_bbox: bool = True, # Normalize targets (l,t,b,r) by stride
                 centerness_on_reg: bool = True, # Centerness is predicted on box regression not classification
                 bbox_coder: str = 'DistancePointBBoxCoder', # it “decodes” the predicted offsets into (x1, y1, x2, y2) coordinates and can also “encode” ground truth boxes into a format the model can learn from.
                ) -> None:
        """
        Docstring for __init__
        
        :param self: Description
        """
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.regression_ranges = regression_ranges
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.bbox_coder = bbox_coder

        self.initialize_head_layers()
    
    def initialize_head_layers(self) -> None:
        """
        Initialize layers of the head
        
        :param self: Description
        """

        # ==== STACKED CONVOLUTIONS (4 layers each) ====
        # These are shared feature extractors before the final predictions

        # Classification convolution tower 4 stacked layers
        self.classification_convs = nn.ModuleList()
        # Classification normalization tower 4 stacked layers
        self.classification_norms = nn.ModuleList()

        # Regression convolution tower 4 stacked layers
        self.regression_convs = nn.ModuleList()
        # Regression normalization tower 4 stacked layers
        self.regression_norms = nn.ModuleList()

        for i in range(self.stacked_convs):
            self.classification_convs.append(
                nn.Conv2d(in_channels=self.in_channels if i == 0 else self.feat_channels,
                      out_channels=self.feat_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False
                )
            )
            self.classification_norms.append(
                nn.GroupNorm(num_groups=32, num_channels=self.feat_channels)
            )

            self.regression_convs.append(
                nn.Conv2d(in_channels=self.in_channels if i == 0 else self.feat_channels,
                      out_channels=self.feat_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False
                )
            )

            self.regression_norms.append(
                nn.GroupNorm(num_groups=32, num_channels=self.feat_channels)
            )

        # ==== FINAL PREDICTION LAYERS ====
        # These produce the actual predictions

        self.conv_centerness = nn.Conv2d(in_channels=self.feat_channels, out_channels=1, kernel_size=3, padding=1) # Convs to learn the centerness and how to define it based on the features
        self.conv_regression = nn.Conv2d(in_channels=self.feat_channels, out_channels=4, kernel_size=3, padding=1) # Convs for bounding box regression. Like to learn the coordinates of bounding boxes
        self.conv_classification = nn.Conv2d(in_channels=self.feat_channels, out_channels=self.num_classes, kernel_size=3, padding=1) # Convs to learn what the object is