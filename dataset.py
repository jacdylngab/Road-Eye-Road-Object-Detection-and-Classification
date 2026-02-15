import torch
import json
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import read_image
import numpy as np

class BDD100KDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        """
        BDD100K dataset for object detection and classification.

        Args:
            images_dir (str or Path): Path to the directory containing images.
            labels_dir (str or Path): Path to the directory containing annotation files.
            transform (callable, optional): Optional transform to be applied
                to each sample (e.g., augmentations, normalization).
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        # Sorting ensures that the images are aligned
        self.images_files = sorted(self.images_dir.glob("*.jpg"))

        self.category_to_idx = {
            "bus": 0,
            "traffic light": 1,
            "traffic sign": 2,
            "person": 3,
            "bike": 4,
            "truck": 5,
            "motor": 6,
            "car": 7,
            "train": 8,
            "rider": 9
        }
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.images_files)

    def __getitem__(self, idx):
        """
            Returns the image and target for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            image (Tensor): The image tensor of shape [C, H, W].
            target (dict): A dictionary containing:
                - "boxes" (Tensor[N, 4]): Bounding boxes in [x1, y1, x2, y2] format.
                - "labels" (Tensor[N]): Class labels for each bounding box.
        """
        # Load images and labels
        image_path = self.images_files[idx]
        labels_path = self.labels_dir / f"{image_path.stem}.json"
        image = read_image(image_path)

        with open(labels_path, "r") as f:
            label_data = json.load(f)

        # Extract objects from the frame
        objects = label_data["frames"][0]["objects"]
        
        # Convert to boxes and labels
        boxes, labels = [], []
        for obj in objects:
            if "box2d" in obj:
                boxes.append([obj["box2d"]["x1"], obj["box2d"]["y1"], obj["box2d"]["x2"], obj["box2d"]["y2"]])
                labels.append(self.category_to_idx[obj["category"]])

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

            # Transform boxes and labels to tensors and then make the target dictionary
            target = {
                "bboxes": boxes, 
                "labels": labels,
                "image_id": torch.tensor([idx]),
                "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)
            }

        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

            # Transform boxes and labels to tensors and then make the target dictionary
            target = {
                "bboxes": boxes, 
                "labels": labels,
                "image_id": torch.tensor([idx]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }

    
        if self.transform:
            # Filter invalid boxes before transformation
            valid_bboxes = []
            valid_labels = []

            for i, box in enumerate(target["bboxes"]):
                x_min, y_min, x_max, y_max = box
                # Check for valid dimensions
                if x_max > x_min and y_max > y_min:
                    valid_bboxes.append(box.tolist()) # Convert tensor to list
                    valid_labels.append(target["labels"][i].item()) # Convert tensor to int
                #else:
                #    print(f"Filtering out invalid box: {box}")

            # Only transform if there are valid boxes
            if len(valid_bboxes) > 0:
                image_np = image.permute(1, 2, 0).numpy() # H x W x C
                transformed = self.transform(
                    image=image_np,
                    bboxes=valid_bboxes,
                    labels=valid_labels
                )

                image = transformed["image"]

                bboxes = transformed["bboxes"]

                # Ensure boxes are always 2D [N, 4] even when there no valid box
                if len(bboxes) == 0:
                    target["bboxes"] = torch.zeros((0, 4), dtype=torch.float32)
                else:
                    target["bboxes"] = torch.tensor(transformed["bboxes"], dtype=torch.float32)

                target["labels"] = torch.tensor(transformed["labels"], dtype=torch.long)

                if target["bboxes"].dim() == 1:
                    target["bboxes"] = target["bboxes"].unsqueeze(0) # [4] -> [1, 4]
                
                # Recompute the area and the iscrowd after the transformation and filtering
                target["area"] = (target["bboxes"][:, 3] - target["bboxes"][:, 1]) * (target["bboxes"][:, 2] - target["bboxes"][:, 0])
                target["iscrowd"] = torch.zeros((len(target["labels"]),), dtype=torch.int64) 
            else:
                # Empty case. No valid bboxes
                target["bboxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros((0,), dtype=torch.long)

                # Recompute the area and the iscrowd after the transformation and filtering
                target["area"] = torch.zeros((0,), dtype=torch.float32) 
                target["iscrowd"] = torch.zeros((0,), dtype=torch.int64) 
        
        return image, target