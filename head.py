from typing import Sequence, Tuple, List, Dict
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss, generalized_box_iou_loss

INF = 1e8

# Type alias for a single image's ground truth annotations
GroundTruth = Dict[str, Tensor]

class Scale(nn.Module):
    """
    A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """
    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

class FCOSHead(nn.Module):
    """
    Anchor-free detection head based on FCOS (Fully Convolutional One-Stage Object Detection).

    FCOS eliminates anchor boxes entirely. Instead, each spatial location on a feature map
    directly predicts:
        - A class distribution (what is the object?)
        - Four distances (l, t, r, b) from the point to the GT box edges
        - A centerness score (how close is this point to the GT box center?)

    Centerness acts as a quality weighting signal during inference — points near object
    centers receive higher centerness scores and produce tighter and better bounding boxes, while
    edge points are suppressed even if their classification scores are high.
    They are suppressed because they produce ambigous bounding boxes.   

    The head is applied independently across all FPN levels. Each level specializes in
    a different scale range, controlled by ``regress_ranges``.

    Based on:
        - Paper:  "FCOS: Fully Convolutional One-Stage Object Detection"
                  (Tian et al., 2019) https://arxiv.org/abs/1904.01355
        - Source: https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/fcos_head.py
    
    FPN feature map (B, C, H, W)
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
    cls tower     reg tower       ← 4× (Conv2d + GroupNorm + ReLU)
        │             │
        ▼             ├──────────────────┐
    conv_cls      conv_reg          conv_ctr
        │             │                 │
        ▼             ▼                 ▼
    cls scores    box distances    centerness

    Args:
        num_classes          (int):            Number of foreground object classes (excluding background).
        in_channels          (int):            Channel width of incoming FPN feature maps.
        feat_channels        (int):            Internal channel width of the conv towers. Default: 256.
        stacked_convs        (int):            Number of conv layers in each tower. Default: 4.
        strides              (Sequence[int]):  Stride of each FPN level in image-space pixels.
                                               Default: (4, 8, 16, 32, 64).
        regress_ranges       (Sequence[Tuple]): Min/max regression distance per FPN level.
                                               Larger ranges → larger objects. Default follows FCOS paper.
        center_sampling      (bool):           If True, only points within a stride-scaled center
                                               region of the GT box are treated as positives.
                                               Reduces noise from ambiguous edge points. Default: True.
                                               Points near the center are positive in the GT others are not if this is true. Helps in training
        center_sample_radius (float):          Radius multiplier for the center sampling region,
                                               relative to the stride of each FPN level. Default: 1.5.

    Example:
        >>> head = FCOSHead(num_classes=10, in_channels=256)
        >>> fpn_outputs = {
        ...     "layer1": torch.randn(2, 256, 90, 160),
        ...     "layer2": torch.randn(2, 256, 45,  80),
        ... }
        >>> cls_scores, bbox_preds, ctr_preds = head(fpn_outputs)
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 feat_channels: int = 256,
                 stacked_convs: int = 4,
                 strides: Sequence[int] = (4, 8, 16, 32, 64),
                 regress_ranges: Sequence[Tuple[int, int | float]] = (
                     (-1, 64),
                     (64, 128),
                     (128, 256),
                     (256, 512),
                     (512, INF)
                ),
                 center_sampling: bool = True, 
                 center_sample_radius: float = 1.5, 
                ) -> None:
        """
        Initialize FCOS head layers, loss functions, and hyperparameters.

        Args:
            num_classes          (int):            Number of foreground classes.
            in_channels          (int):            Input channel count from FPN.
            feat_channels        (int):            Channel width inside the conv towers.
            stacked_convs        (int):            Depth of the classification and regression towers.
            strides              (Sequence[int]):  Per-level FPN strides in image coordinates.
            regress_ranges       (Sequence[Tuple]): Per-level min/max distance for target assignment.
            center_sampling      (bool):           Whether to restrict positives to center regions.
            center_sample_radius (float):          Scaling factor for the center sampling radius.
        
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
        self.loss_centerness = nn.BCEWithLogitsLoss(reduction="sum") # Centerness loss to find the best center on the object. Center points produce better bounding. While edge points produce bad boxes
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in self.strides]) # learnable multiplier that adjusts bounding box predictions for each FPN level.

        self.initialize_head_layers()

    # ------------------------------------------------------------------
    # Layer construction
    # ------------------------------------------------------------------ 

    def initialize_head_layers(self) -> None:
        """
        Build the classification tower, regression tower, and final prediction convs.
        
        Two parallel towers (classification + regression) each contain
        ``stacked_convs`` layers of: Conv2d → GroupNorm → ReLU.
        GroupNorm is used instead of BatchNorm because small batch sizes (common in
        detection) cause BatchNorm statistics to be unreliable.

        Final prediction layers:
            - ``conv_classification``: (feat_channels → num_classes)  per-class logits
            - ``conv_regression``:     (feat_channels → 4)            (l, t, r, b) distances
            - ``conv_centerness``:     (feat_channels → 1)            centerness logit
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

    def forward_fpn_level(self, x: Tensor, stride: int, scale: Scale) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Run the FCOS head on a single FPN feature map.

        Passes the input through the classification and regression towers independently,
        then applies the final prediction convolutions. Centerness is predicted from
        the regression branch, as this was shown to yield better results in the paper.

        During inference, box predictions are rescaled from stride-normalized space
        back to image-space coordinates by multiplying by ``stride``.

        Args:
            x      (Tensor): FPN feature map, shape (B, C, H, W).
            stride (int):    Stride of this FPN level in image-space pixels
                             (e.g. 8 for layer2).
            scale (Scale):   learnable multiplier that adjusts bounding box predictions for each FPN level.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - classification_score    (B, num_classes, H, W): raw class logits
                - bounding_box_prediction (B, 4, H, W):           (l, t, r, b) distances ≥ 0
                - centerness_prediction   (B, 1, H, W):           raw centerness logit 
        """

        classification_feat = x
        regression_feat = x

        # Pass the feature maps from the fpn level to the stacked classification convolutions
        # Classification tower: Conv → GroupNorm → ReLU (per layer)
        for conv, norm in zip(self.classification_convs, self.classification_norms):
            classification_feat = F.relu(norm(conv(classification_feat)))
            #classification_feat = F.leaky_relu(norm(conv(classification_feat)), negative_slope=0.1)

        # Pass the feature maps from the stacked classification convs to the final prediction classification convolution to make final predictions
        classification_score = self.conv_classification(classification_feat)
        
        # Pass the feature maps from the fpn level to the stacked regression convolutions
        # Regression tower: Conv → GroupNorm → ReLU (per layer)
        for conv, norm in zip(self.regression_convs, self.regression_norms):
            #regression_feat = F.relu(norm(conv(regression_feat)))
            regression_feat = norm(conv(regression_feat))

        bounding_box_prediction = self.conv_regression(regression_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bounding_box_prediction = scale(bounding_box_prediction).float()

        # Pass the feature maps from the stacked regression convs to the final regression prediction convolution to make final predictions
        # Box distances must be non-negative — clamp acts as ReLU on the output
        bounding_box_prediction = bounding_box_prediction.clamp(min=0)

        # Centerness is predicted on box regression not classification. The FCOS paper found that this works better.
        centerness_prediction = self.conv_centerness(regression_feat)

        # Normalize targets (l,t,b,r) by stride
        #bounding_box_prediction = bounding_box_prediction.clamp(min=0) # This is the same as ReLU

        # During inference: rescale from stride-space → image-space
        if not self.training:
            bounding_box_prediction *= stride

        return classification_score, bounding_box_prediction, centerness_prediction
    

    def forward(self, featmaps: Dict[str, Tensor]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Run the FCOS head across all FPN levels.

        Iterates over each FPN feature map in ``featmaps`` and calls
        ``forward_fpn_level`` for each, collecting predictions per level.

        Args:
            featmaps (Dict[str, Tensor]): Ordered dict of FPN outputs, e.g.:
                {
                    "layer1": Tensor(B, C, H/4,  W/4),
                    "layer2": Tensor(B, C, H/8,  W/8),
                    "layer3": Tensor(B, C, H/16, W/16),
                    "layer4": Tensor(B, C, H/32, W/32),
                    "layer5": Tensor(B, C, H/64, W/64),
                }

        Returns:
            Tuple[List[Tensor], List[Tensor], List[Tensor]]:
                - classification_scores    (List[Tensor]): one (B, num_classes, H, W) per level
                - bounding_box_predictions (List[Tensor]): one (B, 4, H, W) per level
                - centerness_predictions   (List[Tensor]): one (B, 1, H, W) per level
        """

        # Unpack the dict
        feats = tuple(featmaps.values())

        classification_scores: List[Tensor] = []
        bounding_box_predictions: List[Tensor] = []
        centerness_predictions: List[Tensor] = []

        for feat, stride, scale in zip(feats, self.strides, self.scales):
            classification_score, bounding_box_prediction, centerness_prediction = self.forward_fpn_level(feat, stride, scale)
            classification_scores.append(classification_score)
            bounding_box_predictions.append(bounding_box_prediction)
            centerness_predictions.append(centerness_prediction)
        
        return classification_scores, bounding_box_predictions, centerness_predictions

    # ------------------------------------------------------------------
    # Grid prior generation
    # ------------------------------------------------------------------ 

    def single_level_grid_priors(self,
                                 featmap_size: Tuple[int, int],
                                 level_idx: int,
                                 dtype: torch.dtype = torch.float32,
                                 device: str | torch.device = 'cuda'
                                ) -> Tensor:
        """
        Generate the (x, y) image-space center coordinates for every cell in a
        single FPN feature map level.

        Each cell center is offset by 0.5 before scaling by stride, converting
        from grid indices to image-space coordinates. For example, the top-left
        cell of a stride-8 level maps to image coordinate (4.0, 4.0).

        Args:
            featmap_size (Tuple[int, int]): (H, W) of the feature map.
            level_idx    (int):             Index into ``self.strides`` for this level.
            dtype        (torch.dtype):     Desired output tensor dtype.
            device       (str | device):    Target device.

        Returns:
            Tensor: Grid point coordinates, shape (H × W, 2), each row is [x, y]
                    in image-space pixels.
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
                    featmap_sizes: List[Tuple[int, int]],
                    dtype: torch.dtype = torch.float32,
                    device: str | torch.device = 'cuda'
                    ) -> List[Tensor]:
        """
        Generate image-space grid point coordinates for all FPN levels.

        Calls ``single_level_grid_priors`` for each level and returns the results
        as a list, one entry per FPN level.

        Args:
            featmap_sizes (List[Tuple[int, int]]): (H, W) for each FPN level.
            dtype         (torch.dtype):           Desired tensor dtype.
            device        (str | device):          Target device.

        Returns:
            List[Tensor]: One Tensor per FPN level, each of shape (H_i × W_i, 2).

        Raises:
            AssertionError: If ``len(featmap_sizes) != len(self.strides)``.
        """

        # Sanity check
        assert len(self.strides) == len(featmap_sizes), (
            f"Expected {len(self.strides)} feature map sizes, got {len(featmap_sizes)}"
        )
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
    

    # ------------------------------------------------------------------
    # Target assignment
    #
    # TODO: You need to come back to this and make sure you understand the tensor shapes, bounding box calculations, and the center calculations!
    # YOU NEED TO COME BACK TO THIS CODE AND MAKE SURE YOU UNDERSTAND IT. CAPICHE
    # ------------------------------------------------------------------

    def get_targets_single_image(self,
                                 ground_truth_instances: GroundTruth, 
                                 points: Tensor,
                                 regress_ranges: Tensor,
                                 num_points_per_level: List[int]
                                 ) -> Tuple[Tensor, Tensor]:
        """
        Assign classification labels and regression targets to every grid point
        for a single image.

        For each point, the assignment process is:
            1. Compute (l, t, r, b) distances from the point to every GT box edge.
            2. Optionally restrict positives to center regions (center sampling).
            3. Filter by regression range — each FPN level only accepts objects
               whose max(l,t,r,b) falls within the level's allowed range.
            4. Resolve ambiguity (point inside multiple GT boxes) by choosing the
               GT box with the smallest area.

        Background points (no valid GT assignment) receive label = ``num_classes``.

        Args:
            ground_truth_instances (GroundTruth):  Dict with keys:
                - "bboxes"  (Tensor): shape (G, 4), format [x0, y0, x1, y1]
                - "labels"  (Tensor): shape (G,),   integer class indices
            points              (Tensor): All grid points across all FPN levels,
                                          shape (N, 2), each row [x, y].
            regress_ranges      (Tensor): Per-point regression range,
                                          shape (N, 2), each row [min, max].
            num_points_per_level (List[int]): Number of grid points per FPN level,
                                              used to assign correct strides for
                                              center sampling.

        Returns:
            Tuple[Tensor, Tensor]:
                - labels               (N,):    class index per point, or ``num_classes`` for background
                - bounding_box_targets (N, 4):  (l, t, r, b) distances, or zeros for background

        """

        num_points = points.size(0)
        ground_truth_bounding_boxes = ground_truth_instances["bboxes"]
        ground_truth_labels = ground_truth_instances["labels"]
        num_ground_truths = len(ground_truth_labels)

        # If there no ground truth objects as in no objects to detect
        # The labels are going to be background here the background is 0
        # The bounding box targets are going to be zeros like (0, 0, 0, 0)

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
        x, y = points[:, 0], points[:, 1]

        # Expand x and y 
        x = x[:, None].expand(num_points, num_ground_truths)
        y = y[:, None].expand(num_points, num_ground_truths)

        # Compute (l, t, r, b) distances from each point to each GT box edge
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
            stride = center_x.new_zeros(center_x.shape)

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
            # If the smallest -> 0 -> all are positive which means they are inside
            # Else reject point is not the center region of the GT
            inside_gt_bbox_mask = center_bounding_box.min(-1)[0] > 0

        # Not using center sampling. Like here as long as the box is in ground truth box
        else:
            # If all four distances are positive, the point is inside.
            inside_gt_bbox_mask = bounding_box_targets.min(-1)[0] > 0

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
        labels[min_area == INF] = self.num_classes # Set as Background
        bounding_box_targets = bounding_box_targets[range(num_points), min_aread_inds]

        return labels, bounding_box_targets

    def get_targets(self,
                    points: List[Tensor],
                    batch_ground_truth_instances: List[GroundTruth]
                    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Compute classification labels and regression targets for all images in a batch,
        across all FPN levels.

        Internally calls ``get_targets_single_image`` for each image, then
        reorganizes the results from image-major order to level-major order so
        they can be aligned with the head's per-level predictions during loss
        computation.

        Reshape summary::

            Input  (image-major): [img0[lvl0, lvl1, ...], img1[lvl0, lvl1, ...], ...]
            Output (level-major): [lvl0[img0, img1, ...], lvl1[img0, img1, ...], ...]

        Args:
            points (List[Tensor]): Grid points per FPN level, each of shape (N_i, 2).
            batch_ground_truth_instances (List[GroundTruth]): One entry per image in the batch.

        Returns:
            Tuple[List[Tensor], List[Tensor]]:
                - concat_lvl_labels           (List[Tensor]): per-level label tensors,
                  each of shape (B × N_i,)
                - concat_lvl_bbox_targets     (List[Tensor]): per-level bbox target tensors,
                  each of shape (B × N_i, 4), stride-normalized

        Raises:
            AssertionError: If ``len(points) != len(self.regress_ranges)``.
        """
        
        # Sanity check
        assert len(points) == len(self.regress_ranges), (
            f"Expected {len(self.regress_ranges)} point levels, got {len(points)}"
        )
        num_levels = len(points)

        # Expand regression ranges to align with points at their specific level
        # Each FPN level only predicts objects within a certain size range because different feature map resolutions are good at different object scales.
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]

        # concat all level points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # The number of points per level (fpn level: layer1, layer2, layer3, layer4, layer5)
        num_points = [point.size(0) for point in points]

        # Get the labels and bounding box targets of each image.
        labels_list: List[Tensor] = []
        bounding_box_targets_list: List[Tensor] = []

        for ground_truth_instances in batch_ground_truth_instances:
            # Get targets for this single image (all levels concatenated)
            labels, bounding_box_targets = self.get_targets_single_image(
                ground_truth_instances=ground_truth_instances,
                points=concat_points,
                regress_ranges=concat_regress_ranges,
                num_points_per_level=num_points)
            
            # Split targets by pyramid (fpn) leve level. We do this because loss is computed per level so we have de concatenate (I don't know if that is a word)
            labels_list.append(labels.split(num_points, 0))
            bounding_box_targets_list.append(bounding_box_targets.split(num_points, 0))

        # For training and loss functions we need to go because the detection head outputs predictions per level shaped like: level → (batch × points)
        # So for level i, the model predicts 
        # cls_scores[i]  # shape: [B * n_i, C]
        # bbox_preds[i]  # shape: [B * n_i, 4]
        # To compute loss, targets must match that shape.

        # From: [image0[lvl0, lvl1, lvl2], image1[lvl0, lvl1, lvl2]]
        # To: [lvl0[img0, img1], lvl1[img0, img1], lvl2[img0, img1]]
        # To achieve this we will concatenate per level image

        concat_lvl_labels: List[Tensor] = []
        concat_lv_bounding_box_targets: List[Tensor] = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list])
            )
            bounding_box_targets = torch.cat(
                [bounding_box_targets[i] for bounding_box_targets in bounding_box_targets_list]
            )

            # Normalize bbox targets by stride of this level. Helps in training
            bounding_box_targets = bounding_box_targets / self.strides[i]

            concat_lv_bounding_box_targets.append(bounding_box_targets)
        
        return concat_lvl_labels, concat_lv_bounding_box_targets
    
    # ------------------------------------------------------------------
    # Support methods
    # ------------------------------------------------------------------

    def centerness_target(self, positive_bounding_box_targets: Tensor) -> Tensor:
        """
        Compute centerness targets from (l, t, r, b) regression targets.

        Centerness measures how close a grid point is to the center of its
        assigned GT box. Points at the exact center score 1.0; edge points
        score near 0.0. The score is used to weight the box regression loss
        and to suppress low-quality predictions at inference.

        Formula::

            centerness = sqrt( min(l,r)/max(l,r)  ×  min(t,b)/max(t,b) )

        Args:
            positive_bounding_box_targets (Tensor): Regression targets for positive
                points only, shape (P, 4), each row [l, t, r, b].

        Returns:
            Tensor: Centerness targets in [0, 1], shape (P,).
        """
        # Shape (num_pos, 2) for each
        # Easier to compute min/max along horizontal vs vertical directions
        left_right = positive_bounding_box_targets[:, [0, 2]] # Pick l and r
        top_bottom = positive_bounding_box_targets[:, [1, 3]] # pick t and b

        # If there are no positive points, return something safe to avoid NaNs
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]
            ) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]
            )
        
        return torch.sqrt(centerness_targets)

    def decode_bounding_boxes(self,
                              positive_points: Tensor,
                              positive_bounding_box_distances: Tensor
                              ) -> Tensor:
        """
        Convert predicted (l, t, r, b) distances from grid points into absolute
        (x0, y0, x1, y1) bounding box coordinates.

        This is the inverse of the FCOS regression parameterization:
            x0 = px - l,   y0 = py - t
            x1 = px + r,   y1 = py + b

        Args:
            positive_points                  (Tensor): Grid point centers, shape (P, 2),
                                                       each row [px, py] in image-space.
            positive_bounding_box_distances  (Tensor): Predicted distances, shape (P, 4),
                                                       each row [l, t, r, b].

        Returns:
            Tensor: Decoded boxes in (x0, y0, x1, y1) format, shape (P, 4).
        """
        # Split points
        x, y = positive_points[:, 0], positive_points[:, 1]

        # # Split distances
        l, t, r, b = positive_bounding_box_distances[:, 0], positive_bounding_box_distances[:, 1], \
                     positive_bounding_box_distances[:, 2], positive_bounding_box_distances[:, 3]
        
        # Compute corners
        x0 = x - l
        y0 = y - t
        x1 = x + r 
        y1 = y + b

        # Stack back into a single tensor
        decoded_boxes = torch.stack([x0, y0, x1, y1], dim=-1)

        return decoded_boxes

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self,
             classification_scores: List[Tensor],
             bounding_box_predictions: List[Tensor],
             centerness_predictions: List[Tensor],
             batch_ground_truth_instances: List[GroundTruth]
             ) -> Dict[str, Tensor]:

        """
        Compute the total FCOS training loss for a batch.

        Three losses are computed:
            1. **Classification loss** (Sigmoid Focal Loss): Penalizes wrong class
               predictions. Focal weighting down-weights easy background examples and
               focuses training on hard foreground cases.
            2. **Bounding box loss** (GIoU Loss): Penalizes geometric mismatch between
               predicted and GT boxes. Only computed for positive (foreground) points.
               Weighted by centerness targets to emphasize center-quality predictions.
            3. **Centerness loss** (BCE with Logits): Penalizes inaccurate centerness
               predictions. Only computed for positive points.

        All three losses are normalized by the number of positive points in the batch
        (or by the sum of centerness targets for the box loss) to be independent of
        batch size and image resolution.

        Args:
            classification_scores    (List[Tensor]): Per-level class logits,
                                                     each (B, num_classes, H_i, W_i).
            bounding_box_predictions (List[Tensor]): Per-level (l,t,r,b) predictions,
                                                     each (B, 4, H_i, W_i).
            centerness_predictions   (List[Tensor]): Per-level centerness logits,
                                                     each (B, 1, H_i, W_i).
            batch_ground_truth_instances (List[GroundTruth]): Per-image GT annotations.

        Returns:
            Dict[str, Tensor]: Loss components:
                - "loss_classification": scalar Focal Loss
                - "loss_bounding_box":   scalar GIoU Loss (centerness-weighted)
                - "loss_centerness":     scalar BCE Loss
        """

        # Sanity Check
        assert len(classification_scores) == len(bounding_box_predictions) == len(centerness_predictions), (
            "Mismatch in number of FPN levels across head outputs."
        )

        # ----------------------------------Target assignment---------------------------------
        featmap_sizes = [(featmap.shape[2], featmap.shape[3]) for featmap in classification_scores]
        all_level_points = self.grid_priors(
            featmap_sizes=featmap_sizes,
            dtype=bounding_box_predictions[0].dtype,
            device=bounding_box_predictions[0].device 
        )
        label_targets, bounding_box_targets = self.get_targets(points=all_level_points, 
                                                        batch_ground_truth_instances=batch_ground_truth_instances)

        # We will need the number of images in the batch to flatten the points
        num_imgs = classification_scores[0].size(0) 

        # Flatten classification scores, bounding box predictions, and centerness predictions
        # We flatten because 
        # Loss functions don’t care about FPN levels, images, or grids.
        # They want one long list of predictions and one long list of targets.
        #  For each point in the entire batch:
        #       predict (class, box, centerness)
        # So we need: total_points = sum_over_levels(N × H × W)
        # The shape will be [total_points, C]
        
        flatten_classification_scores = [
            classification_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for classification_score in classification_scores
        ]
        flatten_bounding_box_predictions = [
            bounding_box_prediction.permute(0, 2, 3, 1).reshape(-1, 4) for bounding_box_prediction in bounding_box_predictions
        ]
        flatten_centerness_predictions = [
            centerness_prediction.permute(0, 2, 3, 1).reshape(-1) for centerness_prediction in centerness_predictions
        ]
        flatten_classification_scores = torch.cat(flatten_classification_scores) # (total_N, num_classes)
        flatten_bounding_box_predictions = torch.cat(flatten_bounding_box_predictions)  # (total_N, 4)
        flatten_centerness_predictions = torch.cat(flatten_centerness_predictions) # (total_N,)
        flatten_label_targets = torch.cat(label_targets) # (total_N,)
        flatten_bounding_box_targets = torch.cat(bounding_box_targets) # (total_N, 4)

        # Repeat points to align with bounding box predictions
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points]
        ) # (total_N, 2)

        losses = dict()

        background_index = self.num_classes

        # --- Foreground / background masks ------------------------------------
        # Find foreground points. Foreground == positives
        positive_indices = ((flatten_label_targets >= 0) & (flatten_label_targets < background_index)).nonzero().reshape(-1)
        # Count how many positives there are. Means number of foreground points in this batch
        num_positive = torch.tensor(len(positive_indices), dtype=torch.float, device=bounding_box_predictions[0].device)
        num_positive = max(float(num_positive), 1.0) # Helps avoid division by 0 when doing loss calculations

         # -- Classification loss (all points, fg + bg) ------------------------------------
        # Convert target labels shape to match the input shape and change that to one hot code
        target_one_hot = F.one_hot(
            flatten_label_targets.clamp(max=self.num_classes - 1),
            num_classes=self.num_classes
        ).float()

        # Zero-out background rows
        bg_mask = flatten_label_targets == self.num_classes
        target_one_hot[bg_mask] = 0

        # Classification Loss that answers what the object is. It Downweight easy examples (background) and focus on hard ones
        loss_classification = sigmoid_focal_loss(  # used for multi-label classification
            inputs=flatten_classification_scores,
            targets=target_one_hot, 
            gamma=2.0, # How hard we downweight easy examples
            alpha=0.25, # Balance positives vs negatives
            reduction="sum"
        )

        # Normalize the classification loss by the number of positives
        loss_classification /= num_positive

        # --- Box and centerness losses (positive points only) ----------------------
        # Pick only the positive points
        positive_bounding_box_predictions = flatten_bounding_box_predictions[positive_indices]
        positive_centerness_predictions = flatten_centerness_predictions[positive_indices]
        positive_bounding_box_targets = flatten_bounding_box_targets[positive_indices]
        positive_centerness_targets = self.centerness_target(positive_bounding_box_targets=positive_bounding_box_targets)

        # Compute the normalizing factor. This normalizes the weighted bbox loss
        # pos_centerness_targets.sum() → total weight of positive points
        # detach() → we don’t want this influencing gradients
        # max(..., 1e-6) → avoids division by zero if there are no positives
        centerness_denorm = max(float(positive_centerness_targets.sum().detach()), 1e-6)

        # Only compute the bounding box loss and centerness loss for positive points (inside the GT)
        if len(positive_indices) > 0:
            positive_points = flatten_points[positive_indices]
            positive_decoded_bounding_box_predictions = self.decode_bounding_boxes(positive_points=positive_points, 
                                                                                   positive_bounding_box_distances=positive_bounding_box_predictions)
            positive_decoded_bounding_box_targets = self.decode_bounding_boxes(positive_points=positive_points, 
                                                                                   positive_bounding_box_distances=positive_bounding_box_targets)
            loss_bounding_box = generalized_box_iou_loss(
                positive_decoded_bounding_box_predictions,
                positive_decoded_bounding_box_targets
            )

            loss_bounding_box = (loss_bounding_box * positive_centerness_targets).sum() / centerness_denorm

            loss_centerness = self.loss_centerness(
                positive_centerness_predictions,
                positive_centerness_targets,
            )

            # Normalize the centerness loss by the number of positives
            loss_centerness /= num_positive
        
        else:
            # No positive points
            loss_bounding_box = positive_bounding_box_predictions.sum() # .sum() are Dummy values that don't affect gradients
            loss_centerness = positive_centerness_predictions.sum() # .sum() are Dummy values that don't affect gradients
        
        losses["loss_classification"] = loss_classification
        losses["loss_bounding_box"] = loss_bounding_box
        losses["loss_centerness"] = loss_centerness

        return losses

