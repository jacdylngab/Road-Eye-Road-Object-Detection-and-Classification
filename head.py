from typing import Sequence, Tuple, List, Dict
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss, generalized_box_iou_loss

INF = 1e8

GroundTruth = Dict[str, Tensor]

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
        self.loss_centerness = nn.BCEWithLogitsLoss() # Centerness loss to find the best center on the object. Center points produce better bounding. While edge points produce bad boxes

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
    

    def forward(self, featmaps: Dict[str, Tensor]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Docstring for forward
        
        :param self: Description
        :param featmaps: Features from the upstream network, FPN outputs
        :type x: Tuple[Tensor]
        """

        # Unpack the dict
        feats = tuple(featmaps.values())

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
        # YOU NEED TO COME BACK TO THIS CODE AND MAKE SURE YOU UNDERSTAND IT. CAPICHE

        num_points = points.size(0)
        num_ground_truths = len(ground_truth_instances)
        ground_truth_bounding_boxes = ground_truth_instances["bboxes"]
        ground_truth_labels = ground_truth_instances["labels"]

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
    
    def centerness_target(self, positive_bounding_box_targets: Tensor) -> Tensor:
        """
        Compute centerness targets

        Docstring for centerness_target
        
        :param self: Description
        :param positive_bounding_box_targets: Description
        :type positive_bounding_box_targets: Tensor
        :return: Description
        :rtype: Tensor
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
        Decode predicted distances from points to absolute bounding boxes.

        Args:
            positive_points (Tensor): shape (num_pos, 2), each row [px, py]
            positive_bounding_box_distances (Tensor): shape (num_pos, 4), each row [l, t, r, b]

        Returns:
            Tensor: decoded boxes, shape (num_pos, 4), each row [x0, y0, x1, y1]
        """
        # Split points
        x, y = positive_points[:, 0], positive_points[:, 1]

        # # Split distances
        l, t, r, b = positive_bounding_box_distances[:, 0], positive_bounding_box_distances[: 1], \
                     positive_bounding_box_distances[: 2], positive_bounding_box_distances[:, 3]
        
        # Compute corners
        x0 = x - l
        y0 = y - t
        x1 = x + r 
        y1 = y + b

        # Stack back into a single tensor
        decoded_boxes = torch.stack([x0, y0, x1, y1], dim=-1)

        return decoded_boxes

    def loss(self,
             classification_scores: List[Tensor],
             bounding_box_predictions: List[Tensor],
             centerness_predictions: List[Tensor],
             batch_ground_truth_instances: List[GroundTruth]
             ) -> Dict[str, Tensor]:

        # Calculate the loss based on the features extracted by the detection head

        # Sanity Check
        assert len(classification_scores) == len(bounding_box_predictions) == len(centerness_predictions)
        featmap_sizes = [featmap[-2:] for featmap in classification_scores]
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
        flatten_classification_scores = torch.cat(flatten_classification_scores)
        flatten_bounding_box_predictions = torch.cat(flatten_bounding_box_predictions)
        flatten_centerness_predictions = torch.cat(flatten_centerness_predictions)
        flatten_label_targets = torch.cat(label_targets)
        flatten_bounding_box_targets = torch.cat(bounding_box_targets)

        # Repeat points to align with bounding box predictions
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points]
        )

        losses = dict()

        background_index = self.num_classes

        # Find foreground points. Foreground == positives
        positive_indices = ((flatten_label_targets >= 0) & (flatten_label_targets < background_index)).nonzero().reshape(-1)
        # Count how many positives there are. Means number of foreground points in this batch
        num_positive = torch.tensor(len(positive_indices), dtype=torch.float, device=bounding_box_predictions[0].device)
        num_positive = max(float(num_positive), 1.0) # Helps avoid division by 0 when doing loss calculations

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
                avg_factor=num_positive
            )
        
        else:
            # No positive points
            loss_bounding_box = positive_bounding_box_predictions.sum() # .sum() are Dummy values that don't affect gradients
            loss_centerness = positive_centerness_predictions.sum() # .sum() are Dummy values that don't affect gradients
        
        losses["loss_classification"] = loss_classification
        losses["loss_bounding_box"] = loss_bounding_box
        losses["loss_centerness"] = loss_centerness

        return losses

