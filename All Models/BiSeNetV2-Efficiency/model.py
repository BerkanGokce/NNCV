# BiSeNetV2 - Efficiency

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, act=True):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class StemBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = ConvBNAct(3, 16, kernel_size=3, stride=2, padding=1)

        self.left = nn.Sequential(
            ConvBNAct(16, 8, kernel_size=1, stride=1, padding=0),
            ConvBNAct(8, 16, kernel_size=3, stride=2, padding=1),
        )

        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fuse = ConvBNAct(32, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x_left = self.left(x)
        x_right = self.right(x)
        x = torch.cat([x_left, x_right], dim=1)
        x = self.fuse(x)
        return x


class GELayer(nn.Module):
    def __init__(self, in_channels, out_channels, exp_ratio=6, stride=1):
        super().__init__()
        mid_channels = in_channels * exp_ratio
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = ConvBNAct(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        if stride == 1:
            self.dwconv = ConvBNAct(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels,
            )
            self.pwconv = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.shortcut = nn.Identity()
        else:
            self.dwconv1 = ConvBNAct(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_channels,
            )
            self.dwconv2 = ConvBNAct(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=mid_channels,
            )
            self.pwconv = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.shortcut = nn.Sequential(
                ConvBNAct(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_channels,
                ),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)

        if self.stride == 1:
            x = self.dwconv(x)
            x = self.pwconv(x)
            x = x + self.shortcut(identity)
        else:
            x = self.dwconv1(x)
            x = self.dwconv2(x)
            x = self.pwconv(x)
            x = x + self.shortcut(identity)

        x = self.act(x)
        return x


class CEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.conv_gap = ConvBNAct(channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv_last = ConvBNAct(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity = x
        gap = torch.mean(x, dim=(2, 3), keepdim=True)
        gap = self.bn(gap)
        gap = self.conv_gap(gap)
        x = identity + gap
        x = self.conv_last(x)
        return x


class DetailBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = nn.Sequential(
            ConvBNAct(3, 64, kernel_size=3, stride=2, padding=1),
            ConvBNAct(64, 64, kernel_size=3, stride=1, padding=1),
        )
        self.s2 = nn.Sequential(
            ConvBNAct(64, 64, kernel_size=3, stride=2, padding=1),
            ConvBNAct(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBNAct(64, 64, kernel_size=3, stride=1, padding=1),
        )
        self.s3 = nn.Sequential(
            ConvBNAct(64, 128, kernel_size=3, stride=2, padding=1),
            ConvBNAct(128, 128, kernel_size=3, stride=1, padding=1),
            ConvBNAct(128, 128, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        return x


class SemanticBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = StemBlock()

        self.s3 = nn.Sequential(
            GELayer(16, 32, exp_ratio=6, stride=2),
            GELayer(32, 32, exp_ratio=6, stride=1),
        )

        self.s4 = nn.Sequential(
            GELayer(32, 64, exp_ratio=6, stride=2),
            GELayer(64, 64, exp_ratio=6, stride=1),
        )

        self.s5_4 = nn.Sequential(
            GELayer(64, 128, exp_ratio=6, stride=2),
            GELayer(128, 128, exp_ratio=6, stride=1),
            GELayer(128, 128, exp_ratio=6, stride=1),
            GELayer(128, 128, exp_ratio=6, stride=1),
        )

        self.s5_5 = CEBlock(128)

    def forward(self, x):
        feat2 = self.stem(x)      # 1/4
        feat3 = self.s3(feat2)    # 1/8
        feat4 = self.s4(feat3)    # 1/16
        feat5_4 = self.s5_4(feat4)  # 1/32
        feat5_5 = self.s5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BGALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.detail_proc = nn.Sequential(
            ConvBNAct(128, 128, kernel_size=3, stride=1, padding=1, groups=128),
            ConvBNAct(128, 128, kernel_size=1, stride=1, padding=0),
        )

        self.semantic_proc = ConvBNAct(128, 128, kernel_size=3, stride=1, padding=1)

        self.detail_gate = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Sigmoid(),
        )

        self.semantic_gate = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Sigmoid(),
        )

        self.fuse = ConvBNAct(128, 128, kernel_size=3, stride=1, padding=1)

    def forward(self, detail, semantic):
        semantic_up = F.interpolate(semantic, size=detail.shape[-2:], mode="bilinear", align_corners=False)

        detail_feat = self.detail_proc(detail)
        semantic_feat = self.semantic_proc(semantic_up)

        detail_gate = self.detail_gate(semantic_up)
        semantic_gate = self.semantic_gate(detail)

        out = detail_feat * detail_gate + semantic_feat * semantic_gate
        out = self.fuse(out)
        return out


class SegHead(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.conv = ConvBNAct(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=True)

    def forward(self, x, output_size):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


class Model(nn.Module):
    """
    BiSeNetV2-style semantic segmentation model.

    Returns:
        {
            "out": main segmentation logits,
            "aux2": auxiliary logits from semantic stage 2,
            "aux3": auxiliary logits from semantic stage 3,
            "aux4": auxiliary logits from semantic stage 4,
            "aux5_4": auxiliary logits from semantic stage 5_4,
        }
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 19, pretrained_backbone: bool = False):
        super().__init__()

        if in_channels != 3:
            raise ValueError("BiSeNetV2 expects RGB input, so in_channels must be 3.")

        self.in_channels = in_channels
        self.n_classes = n_classes

        self.detail = DetailBranch()
        self.semantic = SemanticBranch()
        self.bga = BGALayer()

        self.head = SegHead(128, 1024, n_classes)
        self.aux2 = SegHead(16, 128, n_classes)
        self.aux3 = SegHead(32, 128, n_classes)
        self.aux4 = SegHead(64, 128, n_classes)
        self.aux5_4 = SegHead(128, 128, n_classes)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"Expected input with shape (B, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")

        output_size = x.shape[-2:]

        detail = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat5_5 = self.semantic(x)

        fused = self.bga(detail, feat5_5)

        out = self.head(fused, output_size)
        aux2 = self.aux2(feat2, output_size)
        aux3 = self.aux3(feat3, output_size)
        aux4 = self.aux4(feat4, output_size)
        aux5_4 = self.aux5_4(feat5_4, output_size)

        return {
            "out": out,
            "aux2": aux2,
            "aux3": aux3,
            "aux4": aux4,
            "aux5_4": aux5_4,
        }