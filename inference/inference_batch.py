from typing import Dict
import torch
import torchvision.transforms.functional as F
import numpy as np
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.utils import draw_bounding_boxes, save_image
from dataset import BDD100KDataset
from transforms import val_transform
from .inference_utils import (
    INFERENCE_FOLDER_PATH, IMAGES_TEST, LABELS_TEST, BATCH_SIZE, NUM_WORKERS, 
    detection_collate_fn, device, post_proceesing_predictions, denormalize,
    IMAGENET_MEAN, IMAGENET_STD, BDD100K_CLASSES, trained_model 
)

# =============================================================================
# Dataset and DataLoader
# =============================================================================

# Create the folder
INFERENCE_FOLDER_PATH.mkdir(parents=True, exist_ok=True)

# Original dataset
bdd100k_dataset_test = BDD100KDataset(
    images_dir=IMAGES_TEST,
    labels_dir=LABELS_TEST,
    transform=val_transform
)

dataset_size = len(bdd100k_dataset_test)

if dataset_size == 0:
    raise RuntimeError("Dataset is empty - check dataset path on server")

test_loader = torch.utils.data.DataLoader(
    dataset=bdd100k_dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=detection_collate_fn
)

# =============================================================================
# Inference loop
# =============================================================================

pbar =  tqdm(test_loader, desc="Running Inference")

count = 1

metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5, 0.75], class_metrics=True)

metric.reset()

model = trained_model()

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
                img = denormalize(img, mean=np.array(IMAGENET_MEAN), std=np.array(IMAGENET_STD))
                img_uint8 = (img * 255).to(torch.uint8)

                pred = batch_preds[b]
                boxed_image = draw_bounding_boxes(
                    image=img_uint8,
                    boxes=pred["boxes"],
                    labels=[f"{BDD100K_CLASSES.get(l.item(), 'unknown')}: {s:.2f}" for l, s in zip(pred["labels"], pred["scores"])],
                    colors="red",
                    width=2
                )

                boxed_image = F.resize(boxed_image, size=[720, 1280])
                save_path = f"{INFERENCE_FOLDER_PATH}/predicted_image_{count}.png"
                save_image(boxed_image.float() / 255.0, save_path)
                count += 1

# =============================================================================
# Metrics
# =============================================================================

results = metric.compute()

# Per-class mAP aligned with class names
per_class_ap: Dict[str, float] = {
    BDD100K_CLASSES[i]: float(results["map_per_class"][i])
    for i in range(len(BDD100K_CLASSES))
    if i < len(results["map_per_class"])
}

with open("metrics_data.txt", "a") as file:
    file.write(f"mAP: {results["map"]}\n")
    file.write(f"mAP@0.5: {results["map_50"]}\n")
    file.write(f"mAP@0.75: {results["map_75"]}\n")
    file.write(f"Per-class AP:\n")
    for cls_name, ap in per_class_ap.items():
        file.write(f"  {cls_name:<15} {ap:.4f}\n")

print("DONE!")
        
