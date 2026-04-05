import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50

# DeepLabV3
class Model(nn.Module):
    """
    DeepLabV3 model for semantic segmentation on Cityscapes.

    Notes:
    - We keep the class name as `Model` because the submission pipeline expects it.
    - We use a ResNet-50 backbone initialized from ImageNet weights.
    - We do NOT use pretrained segmentation weights on Cityscapes.
    """

    def __init__(
        self,
        in_channels=3,
        n_classes=19,
        pretrained_backbone=True,
    ):
        super().__init__()

        if in_channels != 3:
            raise ValueError("This DeepLabV3 implementation expects RGB input (in_channels=3).")

        backbone_weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None

        self.in_channels = in_channels
        self.n_classes = n_classes

        # torchvision DeepLabV3 with ResNet-50 backbone
        self.model = deeplabv3_resnet50(
            weights=None,                    # do not load pretrained segmentation head
            weights_backbone=backbone_weights,
            num_classes=n_classes,
            aux_loss=False,
        )

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")

        output = self.model(x)

        # torchvision segmentation models return a dict
        # main prediction is stored in output["out"]
        return output["out"]