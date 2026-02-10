from torch import nn, Tensor
from torchvision.models import resnet50, ResNet50_Weights
from backbone import ResnetBackbone
from neck import FPN
from head import FCOSHead

class FCOSDetector(nn.Module):
    # Add type hinting later
    def __init__(self,
                 num_class = 10, 
                 out_channels = 256,
                 pretrained=True
                 ) -> None:
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.backbone = ResnetBackbone(resnet)


        in_channels_list = [256, 512, 1024, 2048]
        self.neck = FPN(in_channels_list=in_channels_list, out_channels=out_channels)

        self.head = FCOSHead(num_classes=num_class, in_channels=out_channels)

    def forward(self,
                images,
                targets=None):
        feat_maps = self.backbone(images)
        fpn_features = self.neck(feat_maps) 
        classification_scores, bounding_box_predictions, centerness_predictions = self.head(fpn_features)

        if self.training and targets is not None:
            losses = self.head.loss(
                classification_scores=classification_scores,
                bounding_box_predictions=bounding_box_predictions,
                centerness_predictions=centerness_predictions,
                batch_ground_truth_instances=targets
            )
            return losses
        else:
            return classification_scores, bounding_box_predictions, centerness_predictions