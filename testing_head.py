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
num_class = 3
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
cls_target = torch.randint(0, num_class, (batch_size, height, width), device=device)
bbox_target = torch.rand(batch_size, 4, height, width, device=device)
centerness_target = torch.rand(batch_size, 1, height, width, device=device)
print(f"cls target: {cls_target}")
print(f"bbox target: {bbox_target}")
print(f"centerness target: {centerness_target}")

bactg_gt_instances = []

# Goal understand this
for b in range(batch_size):
    instance = {
        "labels": cls_target[b].flatten(), # Flatten to list of points
        "bboxes": bbox_target[b].permute(1, 2, 0).reshape(-1, 4), # reshape to [num_points, 4]
        "centerness": centerness_target[b].permute(1, 2, 0).reshape(-1,1) # [num_points, 1]
    }
    bactg_gt_instances.append(instance)

loss_dict = head.loss(
    classification_scores=cls_scores, 
    bounding_box_predictions=bbox_preds, 
    centerness_predictions=centerness_preds,
    batch_ground_truth_instances=bactg_gt_instances)