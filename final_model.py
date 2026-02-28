from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from backbone import ResnetBackbone
from neck import FPN
from head import FCOSHead

class FCOSDetector(nn.Module):
    """
    End-to-end FCOS object detector combining a ResNet-50 backbone,
    FPN neck, and FCOS detection head.

    The model operates in two modes depending on whether ground truth
    annotations are provided:

        - **Training mode** (``targets`` provided): returns a dict of
          scalar losses (classification, box regression, centerness).
        - **Inference mode** (``targets=None``): returns raw per-level
          predictions (class scores, box distances, centerness logits).

    Architecture::

        Input images (B, 3, H, W)
               │
               ▼
        ResNet-50 Backbone  →  {layer1, layer2, layer3, layer4}
               │                (256, 512, 1024, 2048 channels)
               ▼
        FPN Neck            →  {layer1, ..., layer5}
               │                (all 256 channels)
               ▼
        FCOS Head           →  cls_scores, bbox_preds, centerness
               │
               ├─ (training)  → loss dict
               └─ (inference) → raw predictions

    Args:
        num_classes  (int):  Number of foreground object categories. Default: 10.
        out_channels (int):  Uniform channel width across all FPN levels. Default: 256.
        pretrained   (bool): If True, initializes the backbone with ImageNet weights
                             (ResNet50_Weights.IMAGENET1K_V2). Default: True.

    Example:
        >>> model = FCOSDetector(num_classes=10, pretrained=False)
        >>> images = torch.randn(2, 3, 360, 640)
        >>> # Inference
        >>> cls_scores, bbox_preds, ctr_preds = model(images)
        >>> # Training
        >>> targets = [
        ...     {"bboxes": torch.tensor([[10., 20., 100., 200.]]), "labels": torch.tensor([0])},
        ...     {"bboxes": torch.tensor([[50., 60., 150., 250.]]), "labels": torch.tensor([3])},
        ... ]
        >>> losses = model(images, targets=targets)
        >>> losses.keys()
        dict_keys(['loss_classification', 'loss_bounding_box', 'loss_centerness'])
    """
    def __init__(self,
                 num_classes = 10, 
                 out_channels = 256,
                 pretrained=True
                 ) -> None:
        """
        Build the detector by composing backbone, neck, and head.

        Args:
            num_classes  (int):  Number of foreground object classes.
            out_channels (int):  FPN output channel width, passed to both FPN and head.
            pretrained   (bool): Whether to load ImageNet-pretrained backbone weights.
        """
        super().__init__()

        # ==================================== Backbone ========================================
        # ResNet50_Weights.IMAGENET1K_V2 is the modern torchvision weights API.
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.backbone = ResnetBackbone(resnet)

        # ===================================== Neck ===========================================
        # Channel widths for ResNet-50 stages layer1→layer4
        in_channels_list = [256, 512, 1024, 2048]
        self.neck = FPN(in_channels_list=in_channels_list, out_channels=out_channels)

        # ================================ Head ================================================
        self.head = FCOSHead(num_classes=num_classes, in_channels=out_channels)

    def forward(self,
                images,
                targets=None):
        """
        Run the full detection pipeline on a batch of images.

        Args:
            images  (Tensor):                    Batch of preprocessed images,
                                                 shape (B, 3, H, W), normalized
                                                 with ImageNet mean and std.
            targets (Optional[List[GroundTruth]]): Ground truth annotations, one
                                                 dict per image. Each dict must contain:
                                                     - "bboxes"  (Tensor): (G, 4) in [x0,y0,x1,y1]
                                                     - "labels"  (Tensor): (G,) integer class indices
                                                 Pass ``None`` for inference. Default: None.

        Returns:
            - **Training** (``targets`` provided):
              ``Dict[str, Tensor]`` with keys:
                  - ``"loss_classification"`` — Focal Loss over all points
                  - ``"loss_bounding_box"``   — centerness-weighted GIoU Loss
                  - ``"loss_centerness"``     — BCE Loss over positive points

            - **Inference** (``targets=None``):
              ``Tuple[List[Tensor], List[Tensor], List[Tensor]]``:
                  - ``classification_scores``    — per-level (B, num_classes, H_i, W_i)
                  - ``bounding_box_predictions`` — per-level (B, 4, H_i, W_i)
                  - ``centerness_predictions``   — per-level (B, 1, H_i, W_i)

        """
        # ── Backbone: extract multi-scale feature maps ────────────────────────
        feat_maps = self.backbone(images)
        # ── Neck: build feature pyramid with uniform channel width ────────────
        fpn_features = self.neck(feat_maps) 
        # ── Head: predict class scores, box distances, and centerness ─────────
        classification_scores, bounding_box_predictions, centerness_predictions = self.head(fpn_features)

        # ── Training: compute and return losses ───────────────────────────────
        if targets is not None:
            losses = self.head.loss(
                classification_scores=classification_scores,
                bounding_box_predictions=bounding_box_predictions,
                centerness_predictions=centerness_predictions,
                batch_ground_truth_instances=targets
            )
            return losses
        else:
            # ── Inference: return raw predictions for post-processing ─────────────
            return classification_scores, bounding_box_predictions, centerness_predictions