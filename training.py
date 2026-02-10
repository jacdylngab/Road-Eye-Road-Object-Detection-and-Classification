import torch 
from dataset import BDD100KDataset
from transforms import train_transform, val_transform
from final_model import FCOSDetector
from tqdm import tqdm # Progress bar
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =============================================================================
# =============================== Helper Functions ============================
# =============================================================================

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

def train_one_epoch(epoch_index, train_loader, model, optimizer, tensorboard_writer, device):
    model.train() # Sets training mode
    running_loss = 0.0
    last_loss = 0
    skipped = 0
    num_batches = 0

    # Progress bar
    pbar =  tqdm(train_loader, desc=f"Epoch {epoch_index+1}")

    # Think about using optim SGD later
    for batch_idx, (images, targets) in enumerate(pbar):
        # Move images tensor to the GPU
        images = images.to(device)

        # Move all tensors in each target dict to the GPU
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        # Only keep valid images with valid bounding boxes. Skip the invalid ones? 
        valid_indices = []
        for i, t in enumerate(targets):
            if t["bboxes"].numel() > 0:
                valid_indices.append(i)
            else:
                skipped += 1
        if len(valid_indices) == 0:
            continue
        
        images = images[valid_indices]
        targets = [targets[i] for i in valid_indices]

        # Forward pass
        losses = model(images, targets) 
        total_loss = sum(losses.values())

        # Clear old gradients (from previous iteration)
        optimizer.zero_grad()

        # Backward pass - Compute gradients
        total_loss.backward()

        # Optimizer step - Use gradients to update weights
        optimizer.step()

        # Track metrics
        running_loss += total_loss.item()
        running_cls_loss += losses["losses_classification"].item()
        running_bbox_loss += losses["losses_bounding_box"].item()
        running_centerness_loss += losses["losses_centerness"].item()

        # Update progress bar
        pbar.set_postfix({
            "total_loss" : f"{total_loss.item():.3f}",
            "cls_loss": f"{losses["loss_classification"].item():.3f}",
            "bbox_loss": f"{losses["loss_bounding_box"].item():.3f}",
            "center_loss": f"{losses["loss_centerness"].item():.3f}"
        })

        # Log to TensorBoard every N batches
        if batch_idx % 100 == 0:
            # Every 100 batches, do a report
            last_loss = running_loss / 50 # Computes the average loss for the last 50 batches
            global_step = epoch_index * len(train_loader) + batch_idx
            tensorboard_writer.add_scalar("Loss/train_total", total_loss.item(), global_step)
            tensorboard_writer.add_scalar("Loss/train_classification", losses["loss_classification"].item(), global_step)
            tensorboard_writer.add_scalar("Loss/train_bbox", losses["loss_bounding_box"].item(), global_step)
            tensorboard_writer.add_scalar("Loss/train_centerness", losses["loss_centerness"].item(), global_step)

    # Compute epoch averages
    avg_loss = running_loss / num_batches
    avg_cls_loss = running_cls_loss / num_batches
    avg_bbox_loss = running_bbox_loss / num_batches
    avg_centerness_loss = running_centerness_loss / num_batches
    
    print(f"\nEpoch {epoch_index+1} Summary:")
    print(f"  Avg Total Loss: {avg_loss:.4f}")
    print(f"  Avg Cls Loss: {avg_cls_loss:.4f}")
    print(f"  Avg BBox Loss: {avg_bbox_loss:.4f}")
    print(f"  Avg Centerness Loss: {avg_centerness_loss:.4f}")  
    
    print(f"Images Skipped due to invalid bounding boxes: {skipped}")

    return last_loss


images_train = "BDD100K Dataset/bdd100k_images_100k/100k/train"
labels_train = "BDD100K Dataset/bdd100k_labels/100k/train"
images_validation = "BDD100K Dataset/bdd100k_images_100k/100k/val"
labels_validation = "BDD100K Dataset/bdd100k_labels/100k/val"


# Dataset with transforms
bdd100k_dataset_train = BDD100KDataset(images_dir=images_train, labels_dir=labels_train, transform=train_transform)
bdd100k_dataset_validation = BDD100KDataset(images_dir=images_validation, labels_dir=labels_validation, transform=val_transform)

dataset_size = len(bdd100k_dataset_train)

if dataset_size == 0:
    raise RuntimeError("Dataset is empty - check dataset path on server")


# Defining training and validation data loaders
train_loader = torch.utils.data.DataLoader(
    dataset=bdd100k_dataset_train,
    batch_size=2,
    shuffle=True, 
    collate_fn=detection_collate_fn,
    num_workers=2
)

validation_loader = torch.utils.data.DataLoader(
    dataset=bdd100k_dataset_validation,
    batch_size=2,
    shuffle=False,
    collate_fn=detection_collate_fn,
    num_workers=2
)

# Creating and initializing the model
model = FCOSDetector().to(device=device)

# Defining an optimizer algorithm
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
