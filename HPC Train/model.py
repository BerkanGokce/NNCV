import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50

# DeepLabV3 - Fast
class Model(nn.Module):
    """
    DeepLabV3 with a ResNet-50 backbone for Cityscapes semantic segmentation.

    Notes:
    - Keeps the class name `Model` for compatibility with the existing pipeline.
    - Uses ImageNet-pretrained backbone weights when requested.
    - Does not load pretrained segmentation weights.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 19,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()

        if in_channels != 3:
            raise ValueError("This implementation expects RGB input (in_channels=3).")

        backbone_weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None

        self.in_channels = in_channels
        self.n_classes = n_classes

        self.model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=n_classes,
            aux_loss=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")

        output = self.model(x)
        return output["out"]
