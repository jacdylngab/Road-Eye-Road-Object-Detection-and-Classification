from dataset import BDD100KDataset
from transforms import train_transform, val_transform
import matplotlib.pyplot as plt
import torch
import random
import numpy as np

def denormalize(img, mean, std):
    # img: H x W x C, float
    img = img.copy()
    img = img * std + mean # reverse normalization
    img = np.clip(img, 0, 1) # clip to valid range for display
    return img

images_train = "BDD100K Dataset/bdd100k_images_100k/100k/train"
images_labels = "BDD100K Dataset/bdd100k_labels/100k/train"

# Original dataset (no transforms)
bdd100k_dataset_orig = BDD100KDataset(images_dir=images_train, labels_dir=images_labels, transform=None)
# Dataset with transforms
#bdd100k_dataset_trans = BDD100KDataset(images_dir=images_train, labels_dir=images_labels, transform=train_transform)
bdd100k_dataset_trans = BDD100KDataset(images_dir=images_train, labels_dir=images_labels, transform=val_transform)

# Number of images to show
num_images = 4
dataset_size = len(bdd100k_dataset_orig)

# Randomly sample 4 unique indices
indices = random.sample(range(dataset_size), num_images)

fig, axes = plt.subplots(nrows=2, ncols=num_images, figsize=(4*num_images, 8))

for i, idx in enumerate(indices):
    # Original
    image_orig, target_orig = bdd100k_dataset_orig[idx]
    if isinstance(image_orig, torch.Tensor):
        image_orig = image_orig.permute(1, 2, 0).numpy()

    # Transformed
    image_trans, target_trans = bdd100k_dataset_trans[idx]
    if isinstance(image_trans, torch.Tensor):
        image_trans = image_trans.permute(1, 2, 0).numpy()

    # Plot original
    axes[0, i].imshow(image_orig)
    axes[0, i].set_title("Original")
    axes[0, i].axis("off")

    # Plot transformed
    image_display = denormalize(image_trans, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]))
    axes[1, i].imshow(image_display)
    axes[1, i].set_title("Transformed")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()
