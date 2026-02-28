import torch
import torch.nn as nn
from torchvision.models import ResNet
from typing import Dict

"""

Alternative approach using PyTorch's built-in feature extractor utility:

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
    """
    A custom ResNet backbone for multi-scale feature extraction.

    Strips the classification head (avgpool + fc) from a pretrained ResNet
    and exposes intermediate feature maps from all four residual stages.
    These multi-scale outputs are intended for use with a Feature Pyramid
    Network (FPN).

    Architecture overview:
        Input → Stem → Layer1 → Layer2 → Layer3 → Layer4
                         ↓         ↓         ↓         ↓
                      C2 (256)  C3 (512)  C4 (1024) C5 (2048)   ← output channels for ResNet-50
    Example:
        >>> import torchvision.models as models
        >>> resnet = models.resnet50(pretrained=True)
        >>> backbone = ResNetBackbone(resnet)
        >>> x = torch.randn(2, 3, 360, 640)
        >>> features = backbone(x)
        >>> for name, fmap in features.items():
        ...     print(name, fmap.shape)
        layer1 torch.Size([2, 256, 90, 160])
        layer2 torch.Size([2, 512, 45, 80])
        layer3 torch.Size([2, 1024, 23, 40])
        layer4 torch.Size([2, 2048, 12, 20])
    """
    def __init__(self, resnet: ResNet) -> None:
        """
        Initialize the backbone by extracting layers from a pretrained ResNet.

        The fully connected layer and global average pooling are intentionally
        excluded since this backbone is used for dense prediction (detection),
        not classification.

        Args:
            resnet (ResNet): A pretrained ResNet model instance (e.g. from
                             torchvision.models.resnet50(pretrained=True)).
        """
        super().__init__()

        # Stem
        # Initial feature extraction: large receptive field, stride 4 overall
        # Input:  (B, 3,    H,    W)
        # Output: (B, 64, H/4, W/4)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # --- Residual stages ---------------------------------------------
        # Each layer doubles the number of channels and halves spatial dimensions
        self.layer1 = resnet.layer1    # stride 4  → (B,  256, H/4,  W/4)
        self.layer2 = resnet.layer2    # stride 8  → (B,  512, H/8,  W/8)
        self.layer3 = resnet.layer3    # stride 16 → (B, 1024, H/16, W/16)
        self.layer4 = resnet.layer4    # stride 32 → (B, 2048, H/32, W/32)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform a forward pass and return multi-scale feature maps.

        Args:
            x (torch.Tensor): Input image batch of shape (B, 3, H, W).

        Returns:
            Dict[str, torch.Tensor]: A dictionary of feature maps:
                - "layer1": (B,  256, H/4,  W/4)  — fine-grained, high-res
                - "layer2": (B,  512, H/8,  W/8)
                - "layer3": (B, 1024, H/16, W/16)
                - "layer4": (B, 2048, H/32, W/32) — semantic, low-res
        """
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