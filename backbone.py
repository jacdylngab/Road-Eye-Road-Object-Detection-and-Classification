import torch.nn as nn

"""
from torchvision.models import resnet50
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

# Using pytorch create_feature_extractor()
resnet_model = resnet50(pretrained=True)
train_nodes, eval_nodes = get_graph_node_names(resnet_model)

return_nodes = {
    'layer1' : 'layer1',
    'layer2' : 'layer2',
    'layer3' : 'layer3',
    'layer4' : 'layer4',
}

create_feature_extractor(resnet_model, return_nodes=return_nodes)
"""

# Using a custom class. This allows more control
class ResnetBackbone(nn.Module):
    def __init__(self, resnet):
        super().__init__()

        # Stem
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    
    def forward(self, x):
        # Initial convolution (stem) to extract low level feature maps
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Extract multi-scale features
        feature_map1 = self.layer1(x)
        feature_map2 = self.layer2(feature_map1)
        feature_map3 = self.layer3(feature_map2)
        feature_map4 = self.layer4(feature_map3)

        return {
            "layer1": feature_map1,
            "layer2": feature_map2,
            "layer3": feature_map3,
            "layer4": feature_map4
        }

# Small test to make sure the backbone class works.
import torch
from torchvision.models import resnet50, ResNet50_Weights

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
#| ----------- | -------- | ------------ | ---------------------------------- |
# | layer1      | 256      | 56×56        | Early features, high resolution    |
# | layer2      | 512      | 28×28        | Mid-level features                 |
# | layer3      | 1024     | 14×14        | Deeper, more semantic features     |
# | layer4      | 2048     | 7×7          | Very deep features, low resolution |


