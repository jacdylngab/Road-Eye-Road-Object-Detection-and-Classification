import torch
import json
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import read_image
import numpy as np

LABELS_TRAIN = Path("BDD100K Dataset/bdd100k_labels/100k/train")
LABELS_VALIDATION = Path("BDD100K Dataset/bdd100k_labels/100k/val")

unique_labels = set()

print(f"For {LABELS_VALIDATION}")
for item in LABELS_VALIDATION.iterdir():
    if item.is_file():
        with open(item, "r") as f:
            label_data = json.load(f)
        
        # Extract objects from the frame
        objects = label_data["frames"][0]["objects"]
        
        for obj in objects:
            unique_labels.add(obj["category"])

print(unique_labels)