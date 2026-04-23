# DeepLabv3Plus-MobileNet_V3-Efficiency

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = self.pool(x)
        x = self.conv(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates=(12, 24, 36), out_channels: int = 128):
        super().__init__()

        branches = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]

        for rate in atrous_rates:
            branches.append(ASPPConv(in_channels, out_channels, dilation=rate))

        branches.append(ASPPPooling(in_channels, out_channels))

        self.branches = nn.ModuleList(branches)

        self.project = nn.Sequential(
            nn.Conv2d(len(branches) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = [branch(x) for branch in self.branches]
        x = torch.cat(xs, dim=1)
        x = self.project(x)
        return x


class DeepLabV3PlusHead(nn.Module):
    def __init__(self, low_level_channels: int, high_level_channels: int, num_classes: int):
        super().__init__()

        self.aspp = ASPP(high_level_channels, atrous_rates=(12, 24, 36), out_channels=128)

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(low_level_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128 + 32, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, low_level: torch.Tensor, high_level: torch.Tensor, output_size) -> torch.Tensor:
        x = self.aspp(high_level)
        low = self.low_level_proj(low_level)

        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, low], dim=1)
        x = self.decoder(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


class MobileNetV3LargeBackbone(nn.Module):
    def __init__(self, pretrained_backbone: bool):
        super().__init__()

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        backbone = mobilenet_v3_large(weights=weights).features

        # Low-level features: output channels = 24
        self.low_level = backbone[:4]

        # High-level features: final output channels = 960
        self.high_level = backbone[4:]

    def forward(self, x: torch.Tensor):
        low_level = self.low_level(x)
        high_level = self.high_level(low_level)
        return low_level, high_level


class Model(nn.Module):
    """
    Manual DeepLabV3+ with a MobileNetV3-Large backbone.

    Returns:
        {
            "out": main segmentation logits
        }
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 19, pretrained_backbone: bool = False):
        super().__init__()

        if in_channels != 3:
            raise ValueError("DeepLabV3Plus-MobileNetV3Large expects RGB input, so in_channels must be 3.")

        self.in_channels = in_channels
        self.n_classes = n_classes

        self.backbone = MobileNetV3LargeBackbone(pretrained_backbone=pretrained_backbone)
        self.classifier = DeepLabV3PlusHead(
            low_level_channels=24,
            high_level_channels=960,
            num_classes=n_classes,
        )

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"Expected input with shape (B, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")

        input_size = x.shape[-2:]
        low_level, high_level = self.backbone(x)
        out = self.classifier(low_level, high_level, output_size=input_size)

        return {
            "out": out,
        }