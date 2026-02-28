import torch
from torch import Tensor 
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from dataset import BDD100KDataset
from transforms import train_transform, val_transform
from final_model import FCOSDetector
from tqdm import tqdm # Progress bar
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from head import GroundTruth

# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =============================================================================
# Dataset paths
# =============================================================================

IMAGES_TRAIN = "BDD100K Dataset/bdd100k_images_100k/100k/train"
LABELS_TRAIN = "BDD100K Dataset/bdd100k_labels/100k/train"
IMAGES_VALIDATION = "BDD100K Dataset/bdd100k_images_100k/100k/val"
LABELS_VALIDATION = "BDD100K Dataset/bdd100k_labels/100k/val"

# =============================================================================
# Hyperparameters
# =============================================================================


BATCH_SIZE:    int   = 8
NUM_WORKERS:   int   = 12
EPOCHS:        int   = 100
LEARNING_RATE: float = 0.005
MOMENTUM:      float = 0.9
WEIGHT_DECAY:  float = 1e-4
LR_STEP_SIZE:  int   = 30       # StepLR: decay LR every N epochs
LR_GAMMA:      float = 0.1      # StepLR: multiply LR by this on each step
PATIENCE:      int   = 20       # Early stopping: epochs without improvement
MIN_DELTA:     float = 0.001    # Early stopping: minimum meaningful improvement
LOG_INTERVAL:  int   = 100      # TensorBoard batch-level logging frequency

# =============================================================================
# Helper functions
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
# Training and evaluation
# =============================================================================

def train_one_epoch(
        epoch_index:        int,
        train_loader:       DataLoader,
        model:              FCOSDetector,
        optimizer:          torch.optim.Optimizer,
        tensorboard_writer: SummaryWriter,
        device:             torch.device
    ) -> float:
    """
    Run one full training epoch over the training DataLoader.

    For each batch:
        1. Move data to device.
        2. Skip images with empty annotations.
        3. Forward pass → compute losses.
        4. Check for NaN losses (exits on detection).
        5. Backward pass → update weights.
        6. Log batch-level metrics to TensorBoard every ``LOG_INTERVAL`` batches.

    Args:
        epoch_index        (int):              Zero-based epoch index.
        train_loader       (DataLoader):       Training data loader.
        model              (FCOSDetector):     The detector model (must be in train mode).
        optimizer          (Optimizer):        Optimizer managing model parameters.
        tensorboard_writer (SummaryWriter):    TensorBoard writer for logging.
        device             (torch.device):     Target compute device.

    Returns:
        float: Average total loss across all valid batches in this epoch.

    """
    model.train() # Sets training mode
    running_loss = 0.0
    running_cls_loss = 0.0
    running_bbox_loss = 0.0
    running_centerness_loss = 0.0
    skipped = 0
    num_batches = 0

    # Progress bar
    pbar =  tqdm(train_loader, desc=f"epoch {epoch_index+1} Training")

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
        for k, v in losses.items():
            if torch.isnan(v):
                raise RuntimeError(
                    f"NaN detected in '{k}' at epoch {epoch_index + 1}, batch {batch_idx}. "
                    f"Check learning rate, input normalization, and GT box validity."
                )

        total_loss = sum(losses.values())

        # Clear old gradients (from previous iteration)
        optimizer.zero_grad()

        # Backward pass - Compute gradients
        total_loss.backward()

        # Optimizer step - Use gradients to update weights
        optimizer.step()

        # Track losses
        running_loss += total_loss.item()
        running_cls_loss += losses["loss_classification"].item()
        running_bbox_loss += losses["loss_bounding_box"].item()
        running_centerness_loss += losses["loss_centerness"].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            "total_loss" : f"{total_loss.item():.3f}",
            "cls_loss": f"{losses["loss_classification"].item():.3f}",
            "bbox_loss": f"{losses["loss_bounding_box"].item():.3f}",
            "center_loss": f"{losses["loss_centerness"].item():.3f}"
        })

        # Log to TensorBoard every N batches
        if batch_idx % LOG_INTERVAL == 0:
            # Every 100 batches, do a report
            global_step = epoch_index * len(train_loader) + batch_idx
            tensorboard_writer.add_scalar("Loss/train_total", total_loss.item(), global_step)
            tensorboard_writer.add_scalar("Loss/train_classification", losses["loss_classification"].item(), global_step)
            tensorboard_writer.add_scalar("Loss/train_bbox", losses["loss_bounding_box"].item(), global_step)
            tensorboard_writer.add_scalar("Loss/train_centerness", losses["loss_centerness"].item(), global_step)

    # Compute epoch averages
    avg_total_loss = running_loss / num_batches
    avg_cls_loss = running_cls_loss / num_batches
    avg_bbox_loss = running_bbox_loss / num_batches
    avg_centerness_loss = running_centerness_loss / num_batches
    
    print(f"\nEpoch {epoch_index+1} Training Summary:")
    print(f"  Avg Total Loss: {avg_total_loss:.3f}")
    print(f"  Avg Cls Loss: {avg_cls_loss:.3f}")
    print(f"  Avg BBox Loss: {avg_bbox_loss:.3f}")
    print(f"  Avg Centerness Loss: {avg_centerness_loss:.3f}")  
    print(f"  Images Skipped due to invalid bounding boxes: {skipped}")

    # Log epoch averages
    tensorboard_writer.add_scalar("Loss/epoch_avg", avg_total_loss, epoch_index)

    return avg_total_loss

def evaluate(
        epoch_index:        int,
        model:              FCOSDetector,
        validation_loader:  DataLoader,
        tensorboard_writer: SummaryWriter,
        device:             torch.device
        ) -> float:
    """
    Run one full validation epoch over the validation DataLoader.

    Mirrors ``train_one_epoch`` but with gradient computation disabled and no
    weight updates. Loss is still computed by passing targets to the model,
    which triggers the training-mode loss path in ``FCOSDetector.forward``.

    Args:
        epoch_index        (int):           Zero-based epoch index.
        model              (FCOSDetector):  The detector model (set to eval mode internally).
        validation_loader  (DataLoader):    Validation data loader.
        tensorboard_writer (SummaryWriter): TensorBoard writer for logging.
        device             (torch.device):  Target compute device.

    Returns:
        float: Average total validation loss across all valid batches.
    """
    model.eval() # Sets evaluation mode
    running_loss = 0.0
    running_cls_loss = 0.0
    running_bbox_loss = 0.0
    running_centerness_loss = 0.0
    skipped = 0
    num_batches = 0

    # Progress bar
    pbar =  tqdm(validation_loader, desc=f"Epoch {epoch_index+1} Validating")

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
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

            # Track losses
            running_loss += total_loss.item()
            running_cls_loss += losses["loss_classification"].item()
            running_bbox_loss += losses["loss_bounding_box"].item()
            running_centerness_loss += losses["loss_centerness"].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "total_loss" : f"{total_loss.item():.3f}",
                "cls_loss": f"{losses["loss_classification"].item():.3f}",
                "bbox_loss": f"{losses["loss_bounding_box"].item():.3f}",
                "center_loss": f"{losses["loss_centerness"].item():.3f}"
            })

            # Log to TensorBoard every N batches
            if batch_idx % LOG_INTERVAL == 0:
                # Every 100 batches, do a report
                global_step = epoch_index * len(validation_loader) + batch_idx
                tensorboard_writer.add_scalar("Loss/val_total", total_loss.item(), global_step)
                tensorboard_writer.add_scalar("Loss/val_classification", losses["loss_classification"].item(), global_step)
                tensorboard_writer.add_scalar("Loss/val_bbox", losses["loss_bounding_box"].item(), global_step)
                tensorboard_writer.add_scalar("Loss/val_centerness", losses["loss_centerness"].item(), global_step)

    # Compute epoch averages
    avg_total_loss = running_loss / num_batches
    avg_cls_loss = running_cls_loss / num_batches
    avg_bbox_loss = running_bbox_loss / num_batches
    avg_centerness_loss = running_centerness_loss / num_batches
    
    print(f"\nEpoch {epoch_index+1} Validation Summary:")
    print(f"  Avg Total Loss: {avg_total_loss:.3f}")
    print(f"  Avg Cls Loss: {avg_cls_loss:.3f}")
    print(f"  Avg BBox Loss: {avg_bbox_loss:.3f}")
    print(f"  Avg Centerness Loss: {avg_centerness_loss:.3f}")  
    print(f"  Images Skipped due to invalid bounding boxes: {skipped}")

    # Log epoch averages
    tensorboard_writer.add_scalar("Loss/epoch_val_avg", avg_total_loss, epoch_index)

    return avg_total_loss

# =============================================================================
# Dataset and DataLoader setup
# =============================================================================

# Dataset with transforms
bdd100k_dataset_train = BDD100KDataset(
    images_dir=IMAGES_TRAIN,
    labels_dir=LABELS_TRAIN,
    transform=train_transform
)
bdd100k_dataset_validation = BDD100KDataset(
    images_dir=IMAGES_VALIDATION,
    labels_dir=LABELS_VALIDATION,
    transform=val_transform
)

dataset_size = len(bdd100k_dataset_train)

if dataset_size == 0:
    raise RuntimeError("Dataset is empty - check dataset path on server")


# Defining training and validation data loaders
train_loader = torch.utils.data.DataLoader(
    dataset=bdd100k_dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True, 
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=detection_collate_fn
)

validation_loader = torch.utils.data.DataLoader(
    dataset=bdd100k_dataset_validation,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=detection_collate_fn
)

# =============================================================================
# Model, optimizer, and scheduler
# =============================================================================

# Creating and initializing the model
model = FCOSDetector().to(device=device)

# Defining an optimizer algorithm
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer=optimizer,
    step_size=LR_STEP_SIZE,
    gamma=LR_GAMMA
)

# =============================================================================
# TensorBoard
# =============================================================================

# Initialize the summary write from tensorboard for visualizing training. 
# timestamp is just there to make unique folders for every training run, so you don’t overwrite previous logs.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f"runs/roadeye_{timestamp}")

# =============================================================================
# Training loop
# =============================================================================

epoch_no_improve = 0
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    avg_train_total_loss = train_one_epoch(epoch_index=epoch, 
                                     train_loader=train_loader, 
                                     model=model, 
                                     optimizer=optimizer,
                                     tensorboard_writer=writer, 
                                     device=device)
    
    avg_val_total_loss = evaluate(epoch_index=epoch,
                                  model=model,
                                  validation_loader=validation_loader,
                                  tensorboard_writer=writer,
                                  device=device)
    
    # Update lr if needed
    scheduler.step()
    
    # Log the running loss averaged per batch for both training and validation
    writer.add_scalars("Training vs. Validation Loss", 
                      { "Training" : avg_train_total_loss, "Validation" : avg_val_total_loss },
                      epoch + 1)
    writer.flush()

    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"  Train Loss: {avg_train_total_loss:.4f}")
    print(f"  Val Loss:   {avg_val_total_loss:.4f}")
    print(f"  LR:         {scheduler.get_last_lr()[0]:.6f}")

    if avg_val_total_loss < best_val_loss - MIN_DELTA:
        best_val_loss = avg_val_total_loss
        epoch_no_improve = 0
        
        # Saves a full state dict including optimizer and scheduler state, not just model weights. 
        # This means you can resume training from a checkpoint rather than starting over if something interrupts.
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss
            }, "best_model.pt")

        print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")
    else:
        epoch_no_improve += 1
        print(f"  No improvement ({epoch_no_improve}/{PATIENCE})")

    if epoch_no_improve >= PATIENCE:
        print("Early stopping truggered.")
        break

writer.close()
print("Done Training")

