import torch.nn as nn
import torch.nn.functional as F
import torch


class ASPPDecoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.aspp1 = nn.Conv2d(in_channels, 256, kernel_size=1, padding=0, dilation=1)
        self.aspp6 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=6, dilation=6)
        self.aspp12 = nn.Conv2d(
            in_channels, 256, kernel_size=3, padding=12, dilation=12
        )
        self.aspp18 = nn.Conv2d(
            in_channels, 256, kernel_size=3, padding=18, dilation=18
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Conv2d(256 * 5, 256, kernel_size=1)
        self.dropout = nn.Dropout(0.3)
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = F.relu(self.aspp1(x))
        x2 = F.relu(self.aspp6(x))
        x3 = F.relu(self.aspp12(x))
        x4 = F.relu(self.aspp18(x))
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.final_conv(x)
        return x
