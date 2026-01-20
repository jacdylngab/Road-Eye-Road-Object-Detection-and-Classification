import torch
from torchvision.models import resnet50, ResNet50_Weights
from backbone import ResnetBackbone

# Create a pretrained resnet50 model
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Pass the model into the custom backbone
backbone = ResnetBackbone(resnet)

# Create a dummy input
x = torch.randn(1, 3, 224, 224)

# Run forward
outputs = backbone(x)

for name, feat in outputs.items():
    print(name, feat.shape)

# What shapes mean
# | Feature Map | Channels | Spatial Size | Notes                              |
# | ----------- | -------- | ------------ | ---------------------------------- |
# | layer1      | 256      | 56×56        | Early features, high resolution    |
# | layer2      | 512      | 28×28        | Mid-level features                 |
# | layer3      | 1024     | 14×14        | Deeper, more semantic features     |
# | layer4      | 2048     | 7×7          | Very deep features, low resolution |

# Input image     → 224×224 (raw pixels)
# Early layers    → 112×112 (edges, textures)
# Mid layers      → 56×56  (parts)
# Deep layers     → 7×7    (objects)