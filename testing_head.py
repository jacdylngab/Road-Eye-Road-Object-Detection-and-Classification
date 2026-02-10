import torch 
from torchvision.models import resnet50, ResNet50_Weights
from backbone import ResnetBackbone
from neck import FPN
from head import FCOSHead

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Create a pretrained resnet50 model
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Pass the model into the custom backbone
backbone = ResnetBackbone(resnet)
backbone = backbone.to(device=device)

# Create a dummy inputs
num_class = 10 
batch_size = 2 # This means two fake images images
channels = 3
height = 224
width = 224

dummy_images = torch.randn(batch_size, channels, height, width, device=device)

# Run forward
backbone_outputs = backbone(dummy_images)
print("Backbone:")
for name, feat in backbone_outputs.items():
    print(name, feat.shape)

in_channels_list = [feat.shape[1] for feat in backbone_outputs.values()]
#print(f"In channels for FPN: {in_channels_list}")

out_channels = 256 
neck = FPN(in_channels_list=in_channels_list, out_channels=out_channels)
neck = neck.to(device=device)

# Run forward
fpn_outputs = neck(backbone_outputs)
print("\nNeck:")
for name, feat in fpn_outputs.items():
    print(f"{name}: {feat.shape}")

in_channels_list = [feat.shape[1] for feat in fpn_outputs.values()]

head = FCOSHead(num_classes=num_class, in_channels=in_channels_list[0]) 
head = head.to(device=device)

head_outputs = head(fpn_outputs)

cls_scores, bbox_preds, centerness_preds = head_outputs

for i, (cls, bbox, centerness) in enumerate(zip(cls_scores, bbox_preds, centerness_preds)):
    print(f"Layer {i}")
    print(f"Classification scores: {cls.shape}") 
    print(f"Bounding box predictions: {bbox.shape}") 
    print(f"Centerness predictions: {centerness.shape}")

# Dummy targets
batch_ground_truth_instances = []

for img in range(batch_size):
    # Simulate 5 real objects in this image
    num_objects = 5

    # Real bounding boxes: [x_min, y_min, x_max, y_max]
    bboxes = torch.tensor([
        [100, 100, 200, 200],   # Object 1: small box
        [300, 300, 400, 450],   # Object 2: medium box
        [150, 400, 250, 550],   # Object 3: tall box
        [500, 200, 650, 350],   # Object 4: wide box
        [50, 50, 120, 130]      # Object 5: small box
    ], device=device, dtype=torch.float)

    print(f"bboxes: {bboxes}")

    # Real class labels for these 5 objects
    labels = torch.tensor([1, 2, 5, 9, 0], device=device)
    print(f"labels: {labels}")

    instance = {
        "bboxes": bboxes,       # [5, 4] - 5 objects
        "labels": labels        # [5] - 5 labels
    }
    print(f"instance: {instance}")

    batch_ground_truth_instances.append(instance)

loss_dict = head.loss(
    classification_scores=cls_scores, 
    bounding_box_predictions=bbox_preds, 
    centerness_predictions=centerness_preds,
    batch_ground_truth_instances=batch_ground_truth_instances)

print(f"Loss Dict: {loss_dict}")