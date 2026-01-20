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
    """
    Docstring for ResnetBackbone
    """
    def __init__(self, resnet):
        """
        Docstring for __init__
        
        :param self: Description
        :param resnet: Description
        """
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
        """
        Docstring for forward
        
        :param self: Description
        :param x: Description
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