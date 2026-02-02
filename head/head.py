import torch
import torch.nn as nn
from typing import Sequence, Tuple, List, Dict
from torch import Tensor

INF = 1e8

GroundTruth = Dict[str, torch.Tensor]

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
                 regress_ranges: Sequence[Tuple[int, int | float]] = ((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)),
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
        self.regress_ranges = regress_ranges
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
    
    def grid_priors(self, 
                    featmap_sizes: List[Tuple],
                    dtype: torch.dtype = torch.float32,
                    device: str | torch.device = 'cuda') -> List[Tensor]:
        # Generate grid points (image_space/coordinate) of multiple feature levels (fpn levels)

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
    
    def get_targets_single_image(self,
                                 ground_truth_instances: GroundTruth, 
                                 points: Tensor,
                                 regress_ranges: Tensor,
                                 num_points_per_level: List[int]
                                 ) -> Tuple[Tensor, Tensor]:
        # Compute regression, classification, and centerness targets for points in a single images
        # TODO: You need to come back to this and make sure you understand the tensor shapes, bounding box calculations, and the center calculations!
        num_points = points.size(0)
        num_ground_truths = len(ground_truth_instances)
        ground_truth_bounding_boxes = ground_truth_instances["boxes"]
        ground_truth_labels = ground_truth_instances["labels"]

        # If there no ground truth objects as in no objects to detect
        # The labels are going to be background here the background is 0
        # The bounding box targets are going to be zeros lie (0, 0, 0, 0)

        if num_ground_truths == 0:
            return (
                ground_truth_labels.new_full((num_points,), 0),
                ground_truth_bounding_boxes.new_zeros((num_points, 4))
            )

        # Compute the area of the ground truth bounding boxes
        # gt_bboxes[:, 2] - gt_bboxes[:, 0] → width
        # gt_bboxes[:, 3] - gt_bboxes[:, 1] → height
        areas = (ground_truth_bounding_boxes[:, 3] - ground_truth_bounding_boxes[:, 1]) * \
                (ground_truth_bounding_boxes[:, 2] - ground_truth_bounding_boxes[:, 0])
        
        # Repeat areas for every point
        areas = areas[None].repeat(num_points, 1)

        # Expand regression ranges
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_ground_truths, 2)

        # Expand ground truth boxes
        ground_truth_bounding_boxes = ground_truth_bounding_boxes[None].expand(num_points, num_ground_truths, 4)

        # Extract x and y coordinates of points
        x, y = points[:, 0], points[: 1]

        # Expand x and y 
        x = x[:, None].expand(num_points, num_ground_truths)
        y = y[:, None].expand(num_points, num_ground_truths)

        left = x - ground_truth_bounding_boxes[..., 0]
        top = y - ground_truth_bounding_boxes[..., 1]
        right = ground_truth_bounding_boxes[..., 2] - x
        bottom = ground_truth_bounding_boxes[..., 3] - y
        bounding_box_targets = torch.stack((left, top, right, bottom), -1)

        # Condition 1: Center Sampling
        if self.center_sampling:
            radius = self.center_sample_radius

            # Compute the ground truth box centers
            center_x = (ground_truth_bounding_boxes[..., 0] + ground_truth_bounding_boxes[..., 2]) / 2
            center_y = (ground_truth_bounding_boxes[..., 1] + ground_truth_bounding_boxes[..., 3]) / 2

            # Create an empty tensor to store the center bbox coordinates
            center_ground_truths = torch.zeros_like(ground_truth_bounding_boxes)

            # Create empty tensor to store stride values. This builds a stride-scaled center region
            # For example, a point at P5 (Layer 5) represents a bigger area of the image, so its center window must be larger
            stride = center_xs.new_zeros(center_xs.shape)

            # Assign stride per FPN level
            lvl_begin = 0
            for lvl_idx, num_points_level in enumerate(num_points_per_level):
                lvl_end = lvl_begin + num_points_level
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            # Build the center box. This defines a square region around the center of the ground_truth 
            x_mins = center_x - stride
            y_mins = center_y - stride
            x_maxs = center_x + stride
            y_maxs = center_y + stride

            # This clips the center points to make sure they stay in the original GT box
            center_ground_truths[..., 0] = torch.where(x_mins > ground_truth_bounding_boxes[..., 0], x_mins, ground_truth_bounding_boxes[..., 0])
            center_ground_truths[..., 1] = torch.where(y_mins > ground_truth_bounding_boxes[..., 1], y_mins, ground_truth_bounding_boxes[..., 1])
            center_ground_truths[..., 2] = torch.where(x_maxs > ground_truth_bounding_boxes[..., 2], ground_truth_bounding_boxes[..., 2], x_maxs)
            center_ground_truths[..., 3] = torch.where(y_maxs > ground_truth_bounding_boxes[..., 3], ground_truth_bounding_boxes[..., 3], y_maxs)

            # Check if points are inside the center box
            # The code below decides whether a point is inside the center region of a GT box or not

            cb_dist_left = x - center_ground_truths[..., 0]
            cb_dist_top = y - center_ground_truths[..., 1]
            cb_dist_right = center_ground_truths[..., 2] - x
            cb_dist_bottom = center_ground_truths[..., 3] - y

            center_bounding_box = torch.stack((cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)

            # This means: 
            # Take the smallest of (l,t,r,b)
            # If the smallest > 0 -> all are positive which means they are inside
            # Else reject point is not the center region of the GT
            inside_gt_bbox_mask = center_bounding_box.min(-1)[0] > 0

        # Not using center sampling. Like here as long as the box is in ground truth box
        else:
            # If all four distances are positive, the point is inside.
            inside_gt_bbox_mask = bounding_box_targets.mim(-1)[0] > 0

        # Regression range filtering. Limit the regression range for each location
        max_regress_distance = bounding_box_targets.max(-1)[0] 
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1])
        )

        # Resolve conflicts. Multiple GTs per point
        # If there are still more than one objects for a location,
        # Choose the one with minimal area
        # Invalidate GT boxes that failed either condition set them to INF so they won't be selected
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF

        # min_area: [num_points] - the smallest area for each point
        # min_area_inds: [num_points] - which GT index has that smallest area
        min_area, min_aread_inds = areas.min(dim=1)

        # Assign final labels and targets
        labels = ground_truth_labels[min_aread_inds]
        labels[min_area == INF] = 0 # Set as Background
        bounding_box_targets = bounding_box_targets[range(num_points), min_aread_inds]

        return labels, bounding_box_targets

    def get_targets(self,
                    points: List[Tensor],
                    batch_ground_truth_instances: List[GroundTruth]
    ):
        # Gets/computes the features responsible for predicting an object. 
        # Not all objects can predict an object
        # Compute regression, classification, and centerness targets for points in multiple images
        # This function answers for this point, what should the network predict?
        # So for every point:
        # Label = class or background
        # Bounding box target = (l, t, r, b) or zeros
        
        # Sanity check
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)

        # Expand regression ranges to align with points at their specific level
        # Each FPN level only predicts objects within a certain size range because different feature map resolutions are good at different object scales.
        expanded_regress_ranges = [points[i].new_tensor(self.regress_ranges[i])[None].expand_as(points[i] for i in range(num_levels))]

        # concat all level points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # The number of points per level (fpn level: layer1, layer2, layer3, layer4, layer5)
        num_points = [point.size(0) for point in points]

    def loss(self,
             classification_scores: List[Tensor],
             bounding_box_predictions: List[Tensor],
             centerness_predictions: List[Tensor]):

        # Calculate the loss based on the features extracted by the detection head

        # Sanity Check
        assert len(classification_scores) == len(bounding_box_predictions) == len(centerness_predictions)
        featmap_sizes = [featmap[-2:] for featmap in classification_scores]
        all_level_points = self.grid_priors(
            featmap_sizes=featmap_sizes,
            dtype=bounding_box_predictions[0].dtype,
            device=bounding_box_predictions[0].device 
        )
        labels, bounding_box_targets = self.get_targets()
         