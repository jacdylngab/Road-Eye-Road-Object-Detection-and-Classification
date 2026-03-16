from typing import Dict, List, Tuple
from pathlib import Path
import torch
import torchvision
from torch import Tensor
from final_model import FCOSDetector
from head import FCOSHead, GroundTruth

# =============================================================================
# Constants
# =============================================================================

BATCH_SIZE:        int   = 8
NUM_WORKERS:       int   = 12
SCORE_THRESHOLD:   float = 0.5
NMS_IOU_THRESHOLD: float = 0.5
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD:  List[float] = [0.229, 0.224, 0.225]

IMAGES_TEST = "BDD100K Dataset/bdd100k_images_100k/100k/test"
LABELS_TEST = "BDD100K Dataset/bdd100k_labels/100k/test"
PROJECT_ROOT = Path(__file__).resolve().parent.parent # This will point to the project root (parent folder of inference/)
BEST_MODEL_PATH = PROJECT_ROOT / "best_model.pt"
INFERENCE_FOLDER_PATH = Path("inference_imgs") # Folder to store the inference pictures.
METRICS_OUTPUT = Path("metrics_data.txt")

BDD100K_CLASSES: Dict[int, str] = {
            0: "bus",
            1: "traffic light",
            2: "traffic sign",
            3: "person",
            4: "bike",
            5: "truck",
            6: "motor",
            7: "car",
            8: "train",
            9: "rider"
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =============================================================================
# Collate function
# =============================================================================

def detection_collate_fn(
        batch: List[Tuple[Tensor, GroundTruth]]
    ) -> Tuple[Tensor, List[GroundTruth]]:
    """
    Custom collate function for object detection batches.

    Standard ``torch.utils.data.default_collate`` cannot handle variable-length
    target dicts (different numbers of objects per image). This function stacks
    images into a single batch tensor while keeping targets as a plain list.

    Args:
        batch (List[Tuple[Tensor, GroundTruth]]): List of (image, target) pairs
            returned by the dataset's ``__getitem__``. Each target is a dict:
                - "bboxes"  (Tensor): (G, 4) bounding boxes in [x0, y0, x1, y1]
                - "labels"  (Tensor): (G,)   integer class indices

    Returns:
        Tuple[Tensor, List[GroundTruth]]:
            - images  (Tensor):            shape (B, 3, H, W)
            - targets (List[GroundTruth]): list of length B, one dict per image
    """
    images = []
    targets = []

    for sample in batch:
        images.append(sample[0]) # Image
        targets.append(sample[1]) # target (boxes, labels)
    
    # Stack images into a batch tensor
    images = torch.stack(images, dim=0)

    # Keep targets as a list (can't stack due to variable sizes)
    return images, targets

# =============================================================================
# Post-processing
# =============================================================================

def denormalize(
        img:    Tensor, 
        mean:   List[float], 
        std:    List[float]
    ) -> Tensor:
    """
    Reverse ImageNet normalization to restore pixel values to [0, 1].

    Args:
        img  (Tensor):      Normalized image, shape (C, H, W).
        mean (List[float]): Per-channel mean used during normalization.
        std  (List[float]): Per-channel std used during normalization.

    Returns:
        Tensor: Denormalized image in [0, 1], shape (C, H, W).
    """
    # img: H x W x C, float
    mean = torch.tensor(mean, device=img.device).view(-1,1,1)
    std  = torch.tensor(std,  device=img.device).view(-1,1,1)
    img = img * std + mean # reverse normalization
    img = torch.clip(img, 0, 1) # clip to valid range for display
    return img

def classification_post_processing(
        classification_logits:  List[Tensor],
        centerness_logits:      List[Tensor],
        threshold:              float
    ) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Convert raw FCOS classification and centerness logits into thresholded
    per-location class scores and labels across all FPN levels.

    FCOS improves prediction quality by multiplying classification probabilities
    with centerness — this suppresses low-confidence edge predictions that are
    far from GT box centers.

    Processing per FPN level:
        1. Apply sigmoid to classification and centerness logits → probabilities.
        2. Multiply element-wise to form quality-weighted scores.
        3. Flatten spatial dimensions: (B, C, H, W) → (B, H*W, C).
        4. Take argmax over classes: best class + score per location.
        5. Apply score threshold → boolean keep mask.

    Args:
        classification_logits (List[Tensor]): Per-level raw class logits,
                                              each (B, num_classes, H_i, W_i).
        centerness_logits     (List[Tensor]): Per-level raw centerness logits,
                                              each (B, 1, H_i, W_i).
        threshold             (float):        Minimum score to keep a prediction.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - all_scores (B, total_points):        quality-weighted class scores
            - all_labels (B, total_points):        predicted class indices
            - all_keep   (B, total_points) bool:   True where score > threshold
    """
    final_scores = []
    final_labels = []
    final_keep = []

    for cls_logits_lvl, centerness_lvl in zip(classification_logits, centerness_logits):

        # Step 1: Apply sigmoid to change logits into probabilities (0, 1)
        classification_probabilities = torch.sigmoid(cls_logits_lvl)

        # Step 2: Combine with Centerness
        # FCOS improves classification confidence by multiplying with centerness.
        # This reduces low-quality edge predictions
        centerness_probabilities = torch.sigmoid(centerness_lvl)
        classification_scores = classification_probabilities * centerness_probabilities

        # Step 3: Flatten Spatial Locations
        classification_scores = classification_scores.permute(0, 2, 3, 1) # (B, H, W, C)
        B, C, H, W = cls_logits_lvl.shape
        classification_scores = classification_scores.reshape(B, -1, C) # (B, H*W, C)

        # Step 4: Select best class per location
        scores, labels = classification_scores.max(dim=2) # (B, H*W)

        # Step 5: Threshold
        keep = scores > threshold


        # Example it is going to look like
        #scores = [0.92, 0.03, 0.71, 0.15]
        #threshold = 0.5
        #keep = [True, False, True, False]

        final_scores.append(scores)
        final_labels.append(labels)
        final_keep.append(keep)
    
    # Concatenate all levels
    all_scores = torch.cat(final_scores, dim=1)
    all_labels = torch.cat(final_labels, dim=1)
    all_keep = torch.cat(final_keep, dim=1)

    return all_scores, all_labels, all_keep

def bbox_post_processing(
        head:                       FCOSHead,
        bounding_box_predictions:   List[Tensor]
    ) -> Tensor:
    """
    Decode per-level FCOS (l, t, r, b) distance predictions into absolute
    (x0, y0, x1, y1) bounding box coordinates in image space.

    For each FPN level:
        1. Flatten spatial dims: (B, 4, H, W) → (B, H*W, 4).
        2. Generate grid point coordinates for this level.
        3. Decode distances from each grid point: x0=px-l, y0=py-t, x1=px+r, y1=py+b.

    Args:
        head                     (FCOSHead):    The FCOS head, used to access
                                                ``single_level_grid_priors`` and
                                                ``decode_bounding_boxes``.
        bounding_box_predictions (List[Tensor]): Per-level box predictions,
                                                each (B, 4, H_i, W_i).

    Returns:
        Tensor: Decoded boxes in (x0, y0, x1, y1) format,
                shape (B, total_points, 4), in image-space pixels.
    """
    all_decoded_boxes = []

    for lvl_index, bbox_lvl_pred in enumerate(bounding_box_predictions): # lvl_pred: (B, 4, H, W)
        # Step 1: Flatten bounding box predictions
        B, _, H, W = bbox_lvl_pred.shape
        bbox_lvl_pred = bbox_lvl_pred.permute(0, 2, 3, 1) # (B, H, W, 4)
        bbox_lvl_pred = bbox_lvl_pred.reshape(B, -1, 4) # (B, H*W, 4)

        # Step 2: Generate Location coordinates.
        lvl_points = head.single_level_grid_priors(
            featmap_size = (H, W),
            level_idx = lvl_index
        )

        # Step 3: Decoding the boxex. l,t,r,b → x1,y1,x2,y2
        decoded_lvl_boxes = []
        for b in range(B):
            decoded = head.decode_bounding_boxes(positive_points=lvl_points, 
                                                positive_bounding_box_distances=bbox_lvl_pred[b]) # single image
            decoded_lvl_boxes.append(decoded)

        # Stack along batch dimension
        decoded_lvl_boxes = torch.stack(decoded_lvl_boxes, dim=0) # (B, num_points, 4)
        all_decoded_boxes.append(decoded_lvl_boxes)
    
    decoded_boxes = torch.cat(all_decoded_boxes, dim=1)
    
    return decoded_boxes

def multiclass_nms(
        boxes:          Tensor,
        scores:         Tensor,
        labels:         Tensor,
        iou_threshold:  float = NMS_IOU_THRESHOLD
    ) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Apply class-aware Non-Maximum Suppression (NMS) to remove overlapping detections.

    NMS is applied independently per class — a car detection and a truck detection
    at the same location are not suppressed against each other.

    Args:
        boxes         (Tensor): Predicted boxes,  shape (N, 4), format (x0,y0,x1,y1).
        scores        (Tensor): Prediction scores, shape (N,).
        labels        (Tensor): Predicted classes, shape (N,), integer indices.
        iou_threshold (float):  IoU threshold above which the lower-scoring box
                                is suppressed. Default: ``NMS_IOU_THRESHOLD``.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - final_boxes  (M, 4): surviving boxes after NMS
            - final_scores (M,):   corresponding scores
            - final_labels (M,):   corresponding class indices
        where M ≤ N.
    """

    if len(boxes) == 0: # No need to go through nms if there is no boxes
        return boxes, scores, labels
    
    final_boxes = []
    final_scores = []
    final_labels = []

    unique_labels = labels.unique()

    for cls in unique_labels:
        # Get indices for this class
        cls_mask = labels == cls

        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        # Apply NMS for this class only
        # nms() returns indices of the surviving boxes inside cls_boxes.
        keep = torchvision.ops.nms(cls_boxes, cls_scores, iou_threshold)

        final_boxes.append(cls_boxes[keep])
        final_scores.append(cls_scores[keep])
        final_labels.append(
            torch.full((len(keep), ), cls, device=labels.device)
        )

    return (
        torch.cat(final_boxes),
        torch.cat(final_scores),
        torch.cat(final_labels)
    )


def post_proceesing_predictions(
        model: FCOSDetector,
        outputs: Tuple[List[Tensor], List[Tensor], List[Tensor]]
    ) -> List[Dict[str, Tensor]]:
    """
    Full post-processing pipeline: sigmoid → centerness weighting → threshold
    → box decoding → NMS → per-image prediction dicts.

    Args:
        model   (FCOSDetector): The detector, used to access ``model.head``
                                for grid prior generation and box decoding.
        outputs (Tuple):        Raw head outputs from ``model(images)``:
                                    - classification_logits (List[Tensor])
                                    - bounding_box_predictions (List[Tensor])
                                    - centerness_logits (List[Tensor])

    Returns:
        List[Dict[str, Tensor]]: One dict per image in the batch:
            - "boxes"  (M, 4):  final boxes in (x0, y0, x1, y1) image-space
            - "scores" (M,):    confidence scores in (0, 1)
            - "labels" (M,):    integer class indices

        Images with no detections above threshold return empty tensors.
    """
    classification_logits, bounding_box_predictions, centerness_logits = outputs

    all_scores, all_labels, all_keep = classification_post_processing(classification_logits=classification_logits,
                                                                      centerness_logits=centerness_logits, threshold=0.5)

    decoded_boxes = bbox_post_processing(head=model.head, 
                                         bounding_box_predictions=bounding_box_predictions)

    # Split per image
    B = decoded_boxes.shape[0] # batch size 
    batch_preds = []

    # Build per-image predictions
    for b in range(B):
        # Get indices belonging to this image
        keep_b = all_keep[b]
        # Only keep the boxes, scores, and labels that pass the threshold 
        boxes_b = decoded_boxes[b][keep_b]
        scores_b = all_scores[b][keep_b]
        labels_b = all_labels[b][keep_b]

        if boxes_b.numel() == 0:
            batch_preds.append({
                "boxes": torch.empty((0, 4)),
                "scores": torch.empty((0,)),
                "labels": torch.empty((0,), dtype=torch.long)
            })
            continue

        # Apply Non-Maximum Suppression (NMS) to remove overlapping bounding boxes
        final_boxes, final_scores, final_labels = multiclass_nms(
            boxes=boxes_b,
            scores=scores_b,
            labels=labels_b,
            iou_threshold=0.5
        )

        batch_preds.append({
            "boxes": final_boxes.cpu(),
            "scores": final_scores.cpu(),
            "labels": final_labels.cpu()
        })

    return batch_preds

# =============================================================================
# Model
# =============================================================================

def trained_model():
    # Load the trained model
    model = FCOSDetector().to(device=device)

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model