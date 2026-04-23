# ENet - Efficiency

import torch
import torch.nn as nn
import torch.nn.functional as F


class InitialBlock(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 16):
        super().__init__()
        if out_channels < in_channels:
            raise ValueError("out_channels must be >= in_channels in InitialBlock.")

        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.ext_branch = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        out = torch.cat((main, ext), dim=1)
        out = self.batch_norm(out)
        return self.activation(out)


class RegularBottleneck(nn.Module):
    def __init__(
        self,
        channels: int,
        internal_ratio: int = 4,
        kernel_size: int = 3,
        padding: int = 1,
        dilation: int = 1,
        asymmetric: bool = False,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        if internal_ratio <= 1 or internal_ratio > channels:
            raise ValueError("internal_ratio must be in range (1, channels].")

        internal_channels = channels // internal_ratio

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels),
        )

        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    bias=False,
                ),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(internal_channels),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    bias=False,
                ),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(internal_channels),
            )
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(internal_channels),
            )

        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(p=dropout_prob),
        )

        self.out_activation = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.ext_conv1(x)
        out = self.ext_conv2(out)
        out = self.ext_conv3(out)
        out = out + identity
        return self.out_activation(out)


class DownsamplingBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        internal_ratio: int = 4,
        dropout_prob: float = 0.1,
        return_indices: bool = True,
    ):
        super().__init__()

        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise ValueError("internal_ratio must be in range (1, in_channels].")

        internal_channels = in_channels // internal_ratio
        self.out_channels = out_channels
        self.return_indices = return_indices

        self.main_maxpool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            return_indices=return_indices,
        )

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels),
        )

        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels),
        )

        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=dropout_prob),
        )

        self.out_activation = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor):
        if self.return_indices:
            main, indices = self.main_maxpool(x)
        else:
            main = self.main_maxpool(x)
            indices = None

        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)

        n, ch_main, h, w = main.shape
        if ch_main != self.out_channels:
            pad_channels = self.out_channels - ch_main
            padding = torch.zeros(
                n,
                pad_channels,
                h,
                w,
                dtype=main.dtype,
                device=main.device,
            )
            main = torch.cat((main, padding), dim=1)

        out = main + ext
        out = self.out_activation(out)

        if self.return_indices:
            return out, indices
        return out


class UpsamplingBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        internal_ratio: int = 4,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise ValueError("internal_ratio must be in range (1, in_channels].")

        internal_channels = in_channels // internal_ratio

        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.main_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels),
        )

        self.ext_tconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels),
        )

        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=dropout_prob),
        )

        self.out_activation = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor, indices: torch.Tensor, output_size):
        main = self.main_conv(x)
        main = self.main_unpool(main, indices, output_size=output_size)

        ext = self.ext_conv1(x)
        ext = self.ext_tconv2(ext)
        ext = self.ext_conv3(ext)

        out = main + ext
        return self.out_activation(out)


class Model(nn.Module):
    """
    ENet-style semantic segmentation model.

    Returns:
        {
            "out": segmentation logits
        }
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 19, pretrained_backbone: bool = False):
        super().__init__()

        if in_channels != 3:
            raise ValueError("ENet expects RGB input, so in_channels must be 3.")

        self.in_channels = in_channels
        self.n_classes = n_classes

        # Stage 1
        self.initial_block = InitialBlock(in_channels=in_channels, out_channels=16)

        # Stage 2 - encoder
        self.downsample1 = DownsamplingBottleneck(
            in_channels=16,
            out_channels=64,
            dropout_prob=0.01,
            return_indices=True,
        )
        self.reg1_1 = RegularBottleneck(64, dropout_prob=0.01)
        self.reg1_2 = RegularBottleneck(64, dropout_prob=0.01)
        self.reg1_3 = RegularBottleneck(64, dropout_prob=0.01)
        self.reg1_4 = RegularBottleneck(64, dropout_prob=0.01)

        # Stage 3 - encoder
        self.downsample2 = DownsamplingBottleneck(
            in_channels=64,
            out_channels=128,
            dropout_prob=0.1,
            return_indices=True,
        )

        self.reg2_1 = RegularBottleneck(128, dropout_prob=0.1)
        self.reg2_2 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1)
        self.reg2_3 = RegularBottleneck(128, asymmetric=True, kernel_size=5, padding=2, dropout_prob=0.1)
        self.reg2_4 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1)
        self.reg2_5 = RegularBottleneck(128, dropout_prob=0.1)
        self.reg2_6 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1)
        self.reg2_7 = RegularBottleneck(128, asymmetric=True, kernel_size=5, padding=2, dropout_prob=0.1)
        self.reg2_8 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1)

        # Stage 4 - encoder
        self.reg3_1 = RegularBottleneck(128, dropout_prob=0.1)
        self.reg3_2 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1)
        self.reg3_3 = RegularBottleneck(128, asymmetric=True, kernel_size=5, padding=2, dropout_prob=0.1)
        self.reg3_4 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1)
        self.reg3_5 = RegularBottleneck(128, dropout_prob=0.1)
        self.reg3_6 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1)
        self.reg3_7 = RegularBottleneck(128, asymmetric=True, kernel_size=5, padding=2, dropout_prob=0.1)
        self.reg3_8 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1)

        # Stage 5 - decoder
        self.upsample4 = UpsamplingBottleneck(
            in_channels=128,
            out_channels=64,
            dropout_prob=0.1,
        )
        self.reg4_1 = RegularBottleneck(64, dropout_prob=0.1)
        self.reg4_2 = RegularBottleneck(64, dropout_prob=0.1)

        # Stage 6 - decoder
        self.upsample5 = UpsamplingBottleneck(
            in_channels=64,
            out_channels=16,
            dropout_prob=0.1,
        )
        self.reg5_1 = RegularBottleneck(16, dropout_prob=0.1)

        # Final upsampling to original size
        self.fullconv = nn.ConvTranspose2d(
            16,
            n_classes,
            kernel_size=2,
            stride=2,
            bias=True,
        )

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"Expected input with shape (B, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")

        input_size = x.shape

        x = self.initial_block(x)

        stage1_size = x.shape
        x, max_indices1 = self.downsample1(x)
        x = self.reg1_1(x)
        x = self.reg1_2(x)
        x = self.reg1_3(x)
        x = self.reg1_4(x)

        stage2_size = x.shape
        x, max_indices2 = self.downsample2(x)
        x = self.reg2_1(x)
        x = self.reg2_2(x)
        x = self.reg2_3(x)
        x = self.reg2_4(x)
        x = self.reg2_5(x)
        x = self.reg2_6(x)
        x = self.reg2_7(x)
        x = self.reg2_8(x)

        x = self.reg3_1(x)
        x = self.reg3_2(x)
        x = self.reg3_3(x)
        x = self.reg3_4(x)
        x = self.reg3_5(x)
        x = self.reg3_6(x)
        x = self.reg3_7(x)
        x = self.reg3_8(x)

        x = self.upsample4(x, max_indices2, output_size=stage2_size)
        x = self.reg4_1(x)
        x = self.reg4_2(x)

        x = self.upsample5(x, max_indices1, output_size=stage1_size)
        x = self.reg5_1(x)

        out = self.fullconv(x)

        # Safety resize in case of odd-size mismatches
        out = F.interpolate(out, size=input_size[-2:], mode="bilinear", align_corners=False)

        return {"out": out}