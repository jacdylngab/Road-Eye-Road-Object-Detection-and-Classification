import torch
import json
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import read_image

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
            "bus": 1,
            "traffic light": 2,
            "traffic sign": 3,
            "person": 4,
            "bike": 5,
            "truck": 6,
            "motor": 7,
            "car": 8,
            "train": 9,
            "rider": 10
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
        image_path = self.images_dir / self.images_files[idx]
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
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long) 

        # Transform boxes and labels to tensors and then make the target dictionary
        target = {
            "boxes": boxes, 
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)
        }
    
        if self.transform:
            image, target = self.transform(image, target)
        
        return image, target