# DeepLabV3-ResNet50-v2
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50


class Model(nn.Module):
    """DeepLabV3 with a ResNet-50 backbone for Cityscapes segmentation."""

    def __init__(self, in_channels: int = 3, n_classes: int = 19, pretrained_backbone: bool = False):
        super().__init__()
        if in_channels != 3:
            raise ValueError("DeepLabV3-ResNet50 expects RGB input with 3 channels.")

        weights_backbone = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        self.net = deeplabv3_resnet50(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=n_classes,
            aux_loss=True,
        )
        self.in_channels = in_channels
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[1]}")
        return self.net(x)
