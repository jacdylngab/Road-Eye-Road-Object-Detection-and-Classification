from dataset import BDD100KDataset
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes, save_image
import torchvision.transforms.functional as F
from pathlib import Path
from final_model import FCOSDetector
from transforms import val_transform
import numpy as np
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def detection_collate_fn(batch):
    """
    Custom collate function for object detection.
    Handles variable number of objects per image.
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

def denormalize(img, mean, std):
    # img: H x W x C, float
    mean = torch.tensor(mean, device=img.device).view(-1,1,1)
    std  = torch.tensor(std,  device=img.device).view(-1,1,1)
    img = img * std + mean # reverse normalization
    img = torch.clip(img, 0, 1) # clip to valid range for display
    return img

def classification_post_processing(classification_logits, centerness_logits, threshold):
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

def bbox_post_processing(head, bounding_box_predictions):
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

def multiclass_nms(boxes, scores, labels, iou_threshold=0.5):
    """
    Args:
        boxes:  (N, 4)
        scores: (N,)
        labels: (N,)
    Returns:
        final_boxes, final_scores, final_labels
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


def post_proceesing_predictions(model, outputs):
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

images_test = "BDD100K Dataset/bdd100k_images_100k/100k/test"
labels_test = "BDD100K Dataset/bdd100k_labels/100k/test"
best_model_path = Path("best_model.pt")
inference_folder_path = Path("inference_imgs") # Folder to store the inference pictures.

# Create the folder
inference_folder_path.mkdir(parents=True, exist_ok=True)

# Original dataset
bdd100k_dataset_test = BDD100KDataset(images_dir=images_test, labels_dir=labels_test, transform=val_transform)

dataset_size = len(bdd100k_dataset_test)

if dataset_size == 0:
    raise RuntimeError("Dataset is empty - check dataset path on server")

test_loader = torch.utils.data.DataLoader(
    dataset=bdd100k_dataset_test,
    batch_size=4,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=detection_collate_fn
)

# Load the trained model
model = FCOSDetector().to(device=device)
model.load_state_dict(torch.load(best_model_path))
model.to(device)
model.eval()

pbar =  tqdm(test_loader, desc="Running Inference")

classes = {
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

count = 1

metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5, 0.75], class_metrics=True)

metric.reset()
with torch.no_grad():
    for images, targets in pbar:
            # Move images tensor to the GPU
            images = images.to(device)

            # Make predictions
            outputs = model(images)

            batch_preds = post_proceesing_predictions(model=model, outputs=outputs)

            # Move targets to CPU
            targets_CPU = [{"boxes": t["bboxes"].cpu(), "labels": t["labels"].cpu()} for t in targets]

            # Update metric
            metric.update(batch_preds, targets_CPU)

            # Draw bounding boxes
            for b, img in enumerate(images):
                img = denormalize(img, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]))
                img_uint8 = (img * 255).to(torch.uint8)

                pred = batch_preds[b]
                boxed_image = draw_bounding_boxes(
                    image=img_uint8,
                    boxes=pred["boxes"],
                    labels=[f"{classes.get(l.item(), 'unknown')}:{s:.2f}" for l, s in zip(pred["labels"], pred["scores"])],
                    colors="red",
                    width=2
                )

                boxed_image = F.resize(boxed_image, size=[720, 1280])
                save_path = f"{inference_folder_path}/predicted_image_{count}.png"
                save_image(boxed_image.float() / 255.0, save_path)
                count += 1

results = metric.compute()
with open("metrics_data.txt", "w") as file:
    file.write(f"mAP: {results["map"]}\n")
    file.write(f"mAP@0.5: {results["map_50"]}\n")
    file.write(f"mAP@0.75: {results["map_75"]}\n")
    file.write(f"Per-class AP: {results["map_per_class"]}\n")

print("DONE!")
        
