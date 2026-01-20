import torch
from torchvision.models import resnet50, ResNet50_Weights
from backbone import ResnetBackbone
from neck import FPN

# Create a pretrained resnet50 model
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Pass the model into the custom backbone
backbone = ResnetBackbone(resnet)

# Create a dummy input
x = torch.randn(1, 3, 224, 224)

# Run forward
backbone_outputs = backbone(x)
print("Backbone:")
for name, feat in backbone_outputs.items():
    print(name, feat.shape)

in_channels_list = [feat.shape[1] for feat in backbone_outputs.values()]
#print(f"In channels for FPN: {in_channels_list}")

out_channels = 256 
neck = FPN(in_channels_list=in_channels_list, out_channels=out_channels)

# Run forward
fpn_outputs = neck(backbone_outputs)
print("\nNeck:")
for name, feat in fpn_outputs.items():
    print(f"{name}: {feat.shape}")

