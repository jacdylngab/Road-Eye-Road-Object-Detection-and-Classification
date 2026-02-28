from collections import OrderedDict
from typing import Dict, List
import torch.nn.functional as F
from torch import nn, Tensor

class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) neck for multi-scale object detection.

    FPN constructs a rich, multi-scale feature pyramid by combining deep,
    semantically strong features with shallow, spatially precise features
    via a top-down pathway and lateral connections.

    This is particularly effective for detecting objects at varying scales —
    small objects benefit from high-resolution shallow features, while large
    objects benefit from the semantic depth of deeper layers.

    Based on:
        - Paper: "Feature Pyramid Networks for Object Detection"
                 (Lin et al., 2017) https://arxiv.org/abs/1612.03144
        - Source: (Lin et al., 2017) https://github.com/pytorch/vision/blob/main/torchvision/ops/feature_pyramid_network.py 

    Architecture overview (ResNet-50 example with out_channels=256):

        Backbone outputs:          FPN top-down pathway:         Output pyramid:
        ┌─────────────────┐        ┌──────────────────────┐
        │ layer4: 2048ch  │──1x1──►│ P5 (256ch, H/32,W/32)│──3x3──► P5
        │ layer3: 1024ch  │──1x1──►│ P4 (256ch, H/16,W/16)│──3x3──► P4
        │ layer2:  512ch  │──1x1──►│ P3 (256ch, H/8, W/8) │──3x3──► P3
        │ layer1:  256ch  │──1x1──►│ P2 (256ch, H/4, W/4) │──3x3──► P2
        └─────────────────┘        └──────────────────────┘
                                            │ stride-2 conv
                                            ▼
                                           P6 (256ch, H/64,W/64)  ──► layer5

     Args:
        in_channels_list (List[int]): Number of channels for each backbone
                                      feature map, from shallowest to deepest.
                                      E.g. [256, 512, 1024, 2048] for ResNet-50.
        out_channels (int):           Number of output channels for every FPN
                                      level. Typically 256.

    Example:
        >>> backbone_features = {
        ...     "layer1": torch.randn(2, 256,  90, 160),
        ...     "layer2": torch.randn(2, 512,  45,  80),
        ...     "layer3": torch.randn(2, 1024, 23,  40),
        ...     "layer4": torch.randn(2, 2048, 12,  20),
        ... }
        >>> fpn = FPN(in_channels_list=[256, 512, 1024, 2048], out_channels=256)
        >>> output = fpn(backbone_features)
        >>> for name, fmap in output.items():
        ...     print(name, fmap.shape)
        layer1 torch.Size([2, 256, 90, 160])
        layer2 torch.Size([2, 256, 45,  80])
        layer3 torch.Size([2, 256, 23,  40])
        layer4 torch.Size([2, 256, 12,  20])
        layer5 torch.Size([2, 256,  6,  10])
    """
    def __init__(self, in_channels_list: List[int], out_channels: int) -> None:
        """
        Build the FPN by constructing lateral (1x1) and output (3x3) conv layers
        for each backbone stage, plus an extra downsampling conv for P6.

        Two conv lists are constructed in parallel, one entry per backbone stage:
            - ``channel_align_convs``: 1x1 convs that project each backbone stage
              to a uniform ``out_channels`` width (lateral connections).
            - ``output_fpn_convs``:   3x3 convs applied after the top-down fusion
              to smooth aliasing artifacts introduced by upsampling.

        All conv weights are initialized with Kaiming uniform (He init) to keep
        gradient flow stable at the start of training. Biases are zeroed out
        because they are redundant when followed by BatchNorm.

        Args:
            in_channels_list (List[int]): Input channel counts per backbone stage.
                                          Must not contain zeros.
            out_channels (int):           Uniform channel width for all FPN levels.

        Raises:
            ValueError: If any value in ``in_channels_list`` is 0.
        """
        super().__init__()

        self.channel_align_convs = nn.ModuleList()
        self.output_fpn_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels = 0 is not supported")
            
            # Lateral connection: align all backbone stages to out_channels
            # via a 1x1 conv + Batch Normalization (no spatial information is changed here)
            channel_align_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
            # Output conv: smooth the merged feature map after top-down fusion
            # 3x3 conv reduces upsampling artifacts and refines spatial features
            output_fpn_conv = nn.Sequential(
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

            self.channel_align_convs.append(channel_align_conv)
            self.output_fpn_convs.append(output_fpn_conv)

        # Extra pyramid level P6: stride-2 conv on P5 to capture very large objects
        # that may exceed the receptive field of the deepest backbone stage
        self.layer5 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)

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
        Apply the ``idx``-th lateral (1x1) conv to the input tensor.

        This is functionally equivalent to ``self.channel_align_convs[idx](x)``,
        but implemented as a loop to maintain compatibility with TorchScript,
        which does not support direct ``ModuleList`` indexing.

        Args:
            x   (Tensor): Feature map from the backbone, shape (B, C_in, H, W).
            idx (int):    Index of the conv to apply. Supports negative indexing.

        Returns:
            Tensor: Channel-aligned feature map of shape (B, out_channels, H, W).
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
        Apply the ``idx``-th output (3x3) smoothing conv to the input tensor.

        This is functionally equivalent to ``self.output_fpn_convs[idx](x)``,
        but implemented as a loop to maintain compatibility with TorchScript,
        which does not support direct ``ModuleList`` indexing.

        Args:
            x   (Tensor): Fused feature map after top-down addition,
                          shape (B, out_channels, H, W).
            idx (int):    Index of the conv to apply. Supports negative indexing.

        Returns:
            Tensor: Smoothed FPN output of shape (B, out_channels, H, W).
        """
        num_blocks = len(self.output_fpn_convs)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.output_fpn_convs):
            if i == idx:
                out = module(x)
        return out
    
    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Compute the FPN feature pyramid from a dict of backbone feature maps.

        The forward pass has three stages:

        1. **Bottom-up** (already done by the backbone): multi-scale feature maps
           are passed in via ``x``.
        2. **Top-down pathway**: starting from the deepest feature map (most
           semantic), upsample and add to the next shallower lateral feature map,
           progressively enriching spatial resolution with semantic context.
        3. **Extra level (P6)**: a stride-2 conv on the deepest FPN output
           creates an additional low-resolution level for very large objects.

        Args:
            x (Dict[str, Tensor]): Ordered dict of backbone feature maps, e.g.:
                {
                    "layer1": Tensor(B,  256, H/4,  W/4),
                    "layer2": Tensor(B,  512, H/8,  W/8),
                    "layer3": Tensor(B, 1024, H/16, W/16),
                    "layer4": Tensor(B, 2048, H/32, W/32),
                }
        
        Returns:
            Dict[str, Tensor]: Ordered dict of FPN output feature maps, e.g.:
                {
                    "layer1": Tensor(B, 256, H/4,  W/4),   ← P2, small objects
                    "layer2": Tensor(B, 256, H/8,  W/8),   ← P3
                    "layer3": Tensor(B, 256, H/16, W/16),  ← P4
                    "layer4": Tensor(B, 256, H/32, W/32),  ← P5, large objects
                    "layer5": Tensor(B, 256, H/64, W/64),  ← P6, extra level
                }
        """
        # Unpack the dict into two list for easier handling
        names: List[str] = list(x.keys())
        backbone_feature_maps: List[Tensor] = list(x.values())

        # --- Stage 1: Initialize top-down path from the deepest backbone layer ---
        # Apply lateral 1x1 conv to align channels, then smooth with 3x3 conv
        current_top_down = self.get_result_from_channel_align_convs(backbone_feature_maps[-1], -1)
        results = []
        results.append(self.get_result_from_output_fpn_convs(current_top_down, -1))

        # --- Stage 2: Top-down pathway (deep → shallow) -------------------------
        # For each remaining backbone stage (from second-deepest to shallowest):
        #   1. Project the lateral backbone feature to out_channels via 1x1 conv
        #   2. Upsample the current top-down feature to match the lateral resolution
        #   3. Add them element-wise (fusion)
        #   4. Apply 3x3 smoothing conv and prepend to results
        for idx in range(len(backbone_feature_maps) -2, -1, -1):
            # lateral projection of the current backbone feature map via 1x1 conv
            current_lateral = self.get_result_from_channel_align_convs(backbone_feature_maps[idx], idx)
            feat_shape = current_lateral.shape[-2:]  # (H, W) of the current lateral level
            upsampled_top_down = F.interpolate(current_top_down, size=feat_shape, mode="nearest")
            current_top_down = current_lateral + upsampled_top_down
            results.insert(0, self.get_result_from_output_fpn_convs(current_top_down, idx))

        # --- Stage 3: Extra pyramid level (P6) ----------------------------------
        # Stride-2 conv on the deepest FPN output (P5) to extend the pyramid
        # downward for detecting very large objects
        layer5 = self.layer5(results[-1])
        names.append("layer5")
        results.append(layer5)
        
        # Pack results back into an OrderedDict preserving spatial scale order
        out = OrderedDict(zip(names, results))

        return out


