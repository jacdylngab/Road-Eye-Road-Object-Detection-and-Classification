import torch 
from dataset import BDD100KDataset
from transforms import train_transform, val_transform
from final_model import FCOSDetector
from tqdm import tqdm # Progress bar
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

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
    running_cls_loss = 0.0
    running_bbox_loss = 0.0
    running_centerness_loss = 0.0
    skipped = 0
    num_batches = 0

    # Progress bar
    pbar =  tqdm(train_loader, desc=f"epoch {epoch_index+1}")

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
                print(f"NaN detected in {k}")
                exit(1)

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
        if batch_idx % 100 == 0:
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

def evaluate(epoch_index, model, validation_loader, tensorboard_writer, device):
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
            if batch_idx % 100 == 0:
                # Every 100 batches, do a report
                global_step = epoch_index * len(train_loader) + batch_idx
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
    batch_size=8,
    shuffle=True, 
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=detection_collate_fn
)

validation_loader = torch.utils.data.DataLoader(
    dataset=bdd100k_dataset_validation,
    batch_size=8,
    shuffle=False,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=detection_collate_fn
)

# Initialize the summary write from tensorboard for visualizing training. 
# timestamp is just there to make unique folders for every training run, so you don’t overwrite previous logs.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f"runs/roadeye_{timestamp}")

# Creating and initializing the model
model = FCOSDetector().to(device=device)

# Defining an optimizer algorithm
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer=optimizer,
    step_size=30,
    gamma=0.1
)

EPOCHS = 100
patience = 20
epoch_no_improve = 0
best_val_loss = float('inf')
min_delta = 0.001 # minimal improvement to account as "real"

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
    print(f"Training Average Loss: {avg_train_total_loss}")
    print(f"Validation Average Loss: {avg_val_total_loss}")

    if avg_val_total_loss < best_val_loss - min_delta:
        best_val_loss = avg_val_total_loss
        epoch_no_improve = 0
        #model_path = f"best_model_{timestamp}_{epoch}"
        #torch.save(model.state_dict(), model_path)
        torch.save(model.state_dict(), "best_model.pt")
    else:
        epoch_no_improve += 1

    if epoch_no_improve >= patience:
        print("Early stopping truggered.")
        break

print("Done Training")

