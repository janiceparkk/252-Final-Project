import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (Convolution => Batch Normalization => GELU) * 2
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet2D(nn.Module):
    """
    2D U-Net baseline for comparison.
    Flattens time-steps and variables into a 2D feature stack.

    Input: [B, C_in, T_in, H, W]
    Output: [B, C_out, T_out, H, W]
    """

    def __init__(
        self,
        in_chans: int = 8,
        out_chans: int = 8,
        in_len: int = 7,
        out_len: int = 1,
        features: list[int] = [64, 128, 256, 512],
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.in_len = in_len
        self.out_len = out_len

        # Collapse time and variables for 2D convolutions
        input_dim = in_chans * in_len
        output_dim = out_chans * out_len

        # Encode
        self.inc = DoubleConv(input_dim, features[0])
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(features[0], features[1])
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(features[1], features[2])
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(features[2], features[3])
        )

        # Decode
        self.up1 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(features[3], features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(features[2], features[1])

        self.up3 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(features[1], features[0])

        self.outc = nn.Conv2d(features[0], output_dim, kernel_size=1)

    def forward(self, x):
        b, c, t, h, w = x.shape

        # Flatten into a 2D tensor
        x_2d = x.view(b, c * t, h, w)

        # Encoder pass with feature map
        x1 = self.inc(x_2d)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder to handle odd grid sizes (e.g. 241x361)
        x = self.up1(x4)
        x = F.interpolate(x, size=x3.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv_up1(torch.cat([x3, x], dim=1))

        x = self.up2(x)
        x = F.interpolate(x, size=x2.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv_up2(torch.cat([x2, x], dim=1))

        x = self.up3(x)
        x = F.interpolate(x, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv_up3(torch.cat([x1, x], dim=1))

        # Shape: [B, C_out * T_out, H, W]
        logits = self.outc(x)

        # Reshape back to 5D tensor
        out = logits.view(b, self.out_chans, self.out_len, h, w)
        return out
