from collections import OrderedDict
import torch.nn.functional as F
from torch import nn, Tensor

class FPN(nn.Module):
    """
    Docstring for FPN

    Implementing a Feature Pyramid Network. This is based on `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.
    The code is also based on `https://github.com/pytorch/vision/blob/main/torchvision/ops/feature_pyramid_network.py`

    ADD MORE STUFF HERE LATER!
    """
    def __init__(self, in_channels_list: list[int], out_channels: int):
        super().__init__()
        self.channel_align_convs = nn.ModuleList()
        self.output_fpn_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels = 0 is not supported")
            
            # Make all the channels have a uniform number. All the layers are going to have the same channel numbers
            channel_align_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
            # This is for smoothing out the layers.
            output_fpn_conv = nn.Sequential(
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

            self.channel_align_convs.append(channel_align_conv)
            self.output_fpn_convs.append(output_fpn_conv)

        # Initialize all convolution weights and biases in the FPN/neck
        # - We use Kaiming uniform initialization for weights to keep signals stable
        #   and help training converge faster.
        # - Biases are set to zero because they are redundant when followed by BatchNorm.
        # - This is done here to ensure all conv layers start well-behaved,
        #   without affecting any separately initialized top_blocks (if present).
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
    def get_result_from_channel_align_convs(self, x: Tensor, idx: int) -> Tensor:
        """
        Docstring for get_result_from_channel_align_convs
        
        :param self: Description
        :param x: Description
        :type x: Tensor
        :param idx: Description
        :type idx: int
        :return: Description
        :rtype: Tensor

        This is equivalent to self.channel_align_convs[idx](x).
        This is done because torchscript doesn't support this yet
        """
        num_blocks = len(self.channel_align_convs)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.channel_align_convs):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_output_fpn_convs(self, x: Tensor, idx: int) -> Tensor:
        """
        Docstring for get_result_from_output_fpn_convs
        
        :param self: Description
        :param x: Description
        :type x: Tensor
        :param idx: Description
        :type idx: int
        :return: Description
        :rtype: Tensor
         
        This is equivalent to self.output_fpn_convs[idx](x).
        This is done because torchscript doesn't support this yet
        """
        num_blocks = len(self.output_fpn_convs)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.output_fpn_convs):
            if i == idx:
                out = module(x)
        return out
    
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Docstring for forward
        
        :param self: Description
        :param x: Description
        :type x: dict[str, Tensor]
        :return: Description
        :rtype: dict[str, Tensor]
        """
        # Unpack the dict into two list for easier handling
        names = list(x.keys())
        backbone_feature_maps = list(x.values())

        # Start with the very top (deep layer) from the backbone
        current_top_down = self.get_result_from_channel_align_convs(backbone_feature_maps[-1], -1)
        results = []
        results.append(self.get_result_from_output_fpn_convs(current_top_down, -1))

        # Go down (top_down)
        for idx in range(len(backbone_feature_maps) -2, -1, -1):
            # lateral projection of the current backbone feature map via 1x1 conv
            current_lateral = self.get_result_from_channel_align_convs(backbone_feature_maps[idx], idx)
            feat_shape = current_lateral.shape[-2:]
            upsampled_top_down = F.interpolate(current_top_down, size=feat_shape, mode="nearest")
            current_top_down = current_lateral + upsampled_top_down
            results.insert(0, self.get_result_from_output_fpn_convs(current_top_down, idx))
        
        # Transform the results into an ordered dict
        out = OrderedDict(zip(names, results))

        return out


