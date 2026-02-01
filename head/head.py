import torch
import torch.nn as nn
from typing import Sequence, Tuple, List
from torch import Tensor

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

    def forward_fpn_level(self, x: Tensor, stride: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward features of a single fpn level

        Docstring for forward_fpn_level
        
        :param self: Description
        :param x: Description
        :type x: Tensor
        :return: Description
        :rtype: Tuple[Tensor, Tensor, Tensor]
        """

        
        classification_feat = x
        regression_feat = x

        # Pass the feature maps from the fpn level to the stacked classification convolutions
        for classification_layer in self.classification_convs:
            classification_feat = classification_layer(classification_feat)

        # Pass the feature maps from the stacked classification convs to the final prediction classification convolution to make final predictions
        classification_score = self.conv_classification(classification_feat)
        
        # Pass the feature maps from the fpn level to the stacked regression convolutions
        for regression_layer in self.regression_convs:
            regression_feat = regression_layer(regression_feat)

        # Pass the feature maps from the stacked regression convs to the final regression prediction convolution to make final predictions
        bounding_box_prediction = self.conv_regression(regression_feat)

        # Centerness is predicted on box regression not classification. The FCOS paper found that this works better.
        centerness_prediction = self.conv_centerness(regression_feat)

        # Normalize targets (l,t,b,r) by stride
        bounding_box_prediction = bounding_box_prediction.clamp(min=0) # This is the same as ReLU

        if not self.training:
            bounding_box_prediction *= stride

        return classification_score, bounding_box_prediction, centerness_prediction
    

    def forward(self, feats: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Docstring for forward
        
        :param self: Description
        :param feats: Features from the upstream network, each is
                a 4D-tensor.
        :type x: Tuple[Tensor]
        """

        classification_scores: List[Tensor] = []
        bounding_box_predictions: List[Tensor] = []
        centerness_predictions: List[Tensor] = []

        for feat, stride in zip(feats, self.strides):
            classification_score, bounding_box_prediction, centerness_prediction = self.forward_fpn_level(feat, stride)
            classification_scores.append(classification_score)
            bounding_box_predictions.append(bounding_box_prediction)
            centerness_predictions.append(centerness_prediction)
        
        return classification_scores, bounding_box_predictions, centerness_predictions
    
    def single_level_grid_priors(self,
                                 featmap_size: Tuple[int],
                                 level_idx: int,
                                 dtype: torch.dtype = torch.float32,
                                 device: str | torch.device = 'cuda') -> Tensor:
        """
        Generate grid points of a single level
        Docstring for single_level_grid_priors
        
        :param self: Description
        :param featmap_size: Description
        :type featmap_size: Tuple
        :param level_idx: Description
        :type level_idx: int
        :param dtype: Description
        :type dtype: torch.dtype
        :param device: Description
        :type device: Device
        :return: Description
        :rtype: Tensor
        """
        feat_h, feat_w = featmap_size
        stride = self.strides[level_idx]

        # We add offset because we want cell centers not cell corners. FCOS predicts using cell centers
        # We multiply by stride to change from feature map coordinates to image coordinates
        x = (torch.arange(0, feat_w, device=device) + 0.5) * stride
        x = x.to(dtype) # Keep featmap_size as a Tensor

        y = (torch.arange(0, feat_h, device=device) + 0.5) * stride
        y = y.to(dtype) # Keep featmap_size as a Tensor

        yy, xx = torch.meshgrid(y, x, indexing='ij')

        points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

        all_points = points.to(device)
        return all_points
    
    def grid_priors(self, featmap_sizes: List[Tuple],
                     dtype: torch.dtype = torch.float32,
                     device: str | torch.device = 'cuda') -> List[Tensor]:
        # Generate grid points of multiple feature levels (fpn levels)
        # Sanity check
        assert len(self.strides) == len(featmap_sizes)
        multi_level_priors: List[Tensor] = []
        for i in range(len(self.strides)):
            priors = self.single_level_grid_priors(
                featmap_sizes[i],
                level_idx=i,
                dtype=dtype,
                device=device
            )
            multi_level_priors.append(priors)

        return multi_level_priors
    
    def get_targets(self):
        pass
       
    def loss_calculations(self,
                          classification_scores: List[Tensor],
                          bounding_box_predictions: List[Tensor],
                          centerness_predictions: List[Tensor]):
        # Sanity Check
        assert len(classification_scores) == len(bounding_box_predictions) == len(centerness_predictions)
        featmap_sizes = [featmap[-2:] for featmap in classification_scores]
        all_level_points = self.grid_priors(
            featmap_sizes=featmap_sizes,
            dtype=bounding_box_predictions[0].dtype,
            device=bounding_box_predictions[0].device 
        )
        lab
         