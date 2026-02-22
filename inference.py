from dataset import BDD100KDataset
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes
from pathlib import Path
from final_model import FCOSDetector
from transforms import val_transform
from tqdm import tqdm # Progress bar
import numpy as np

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

    #for lvl in range(len(classification_logits)):
        #cls_logits_lvl = classification_logits[lvl]   # (B, C, H, W)
        #centerness_lvl = centerness_logits[lvl]       # (B, 1, H, W)

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

    # Only keep the boxes, scores, and labels that pass the threshold 
    best_bboxes = decoded_boxes[all_keep]
    best_scores = all_scores[all_keep]
    best_labels = all_labels[all_keep]

    # Apply Non-Maximum Suppression (NMS) to remove overlapping bounding boxes
    final_boxes, final_scores, final_labels = multiclass_nms(
        boxes=best_bboxes,
        scores=best_scores,
        labels=best_labels,
        iou_threshold=0.5
    )

    return final_boxes, final_scores, final_labels

images_test = "BDD100K Dataset/bdd100k_images_100k/100k/test"
labels_test = "BDD100K Dataset/bdd100k_labels/100k/test"
best_model_path = Path("best_model.pt")
inference_folder_path = Path() # Folder to store the inference pictures.

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

#pbar =  tqdm(test_loader)
count = 0
with torch.no_grad():
    for images, targets in test_loader:
            if count == 1: # Just one batch for now
                 break
            # Move images tensor to the GPU
            images = images.to(device)

            # Make predictions
            outputs = model(images)
            #print(outputs)
            final_boxes, final_scores, final_labels = post_proceesing_predictions(model=model, outputs=outputs)

            # Draw bounding boxes
            for img in images:
                img = denormalize(img, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]))
                img_uint8 = (img * 255).to(torch.uint8)

                boxed_image = draw_bounding_boxes(
                    image=img_uint8,
                    boxes=final_boxes,
                    labels = [f"{l.item()}:{s:.2f}" for l, s in zip(final_labels, final_scores)],
                    width=2
                )

                plt.imshow(boxed_image.permute(1,2,0).cpu().numpy())  # H x W x C for plt
                plt.savefig("predicted_image.png")
                plt.axis("off")

            count += 1
