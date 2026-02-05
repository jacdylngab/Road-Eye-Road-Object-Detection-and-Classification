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

head = FCOSHead(num_classes=3, in_channels=in_channels_list) 
head = head.to(device=device)

head_outputs = head(fpn_outputs)

cls_scores, bbox_preds, centerness_preds = head_outputs

for i, (cls, bbox, centerness) in enumerate(zip(cls_scores, bbox_preds, centerness_preds)):
    print(f"Layer {i}")
    print(f"Classification scores: {cls.shape}") 
    print(f"Bounding box predictions: {bbox.shape}") 
    print(f"Centerness predictions: {centerness.shape}") 