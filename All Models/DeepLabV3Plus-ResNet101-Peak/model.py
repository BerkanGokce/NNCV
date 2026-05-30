# DeepLabv3Plus-Resnet101-Peak Performance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet101_Weights, resnet101


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
    def __init__(self, in_channels: int, atrous_rates=(12, 24, 36), out_channels: int = 256):
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

        self.aspp = ASPP(high_level_channels, atrous_rates=(12, 24, 36), out_channels=256)

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, low_level: torch.Tensor, high_level: torch.Tensor, output_size) -> torch.Tensor:
        x = self.aspp(high_level)
        low = self.low_level_proj(low_level)

        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, low], dim=1)
        x = self.decoder(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


class AuxHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )


class ResNet101Backbone(nn.Module):
    def __init__(self, pretrained_backbone: bool):
        super().__init__()

        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        backbone = resnet101(
            weights=weights,
            replace_stride_with_dilation=[False, True, True],
        )

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1   # low-level
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3   # aux
        self.layer4 = backbone.layer4   # high-level

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        low_level = self.layer1(x)
        x = self.layer2(low_level)
        aux = self.layer3(x)
        high_level = self.layer4(aux)

        return low_level, aux, high_level


class Model(nn.Module):
    """
    Manual DeepLabV3+ with a ResNet-101 backbone.

    Returns:
        {
            "out": main segmentation logits,
            "aux": auxiliary segmentation logits
        }
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 19, pretrained_backbone: bool = False):
        super().__init__()

        if in_channels != 3:
            raise ValueError("DeepLabV3Plus-ResNet101 expects RGB input, so in_channels must be 3.")

        self.in_channels = in_channels
        self.n_classes = n_classes

        self.backbone = ResNet101Backbone(pretrained_backbone=pretrained_backbone)
        self.classifier = DeepLabV3PlusHead(
            low_level_channels=256,
            high_level_channels=2048,
            num_classes=n_classes,
        )
        self.aux_classifier = AuxHead(in_channels=1024, num_classes=n_classes)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"Expected input with shape (B, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")

        input_size = x.shape[-2:]
        low_level, aux_features, high_level = self.backbone(x)

        out = self.classifier(low_level, high_level, output_size=input_size)
        aux = self.aux_classifier(aux_features)
        aux = F.interpolate(aux, size=input_size, mode="bilinear", align_corners=False)

        return {
            "out": out,
            "aux": aux,
        }