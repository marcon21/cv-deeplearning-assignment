import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_base import ModelBase
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class UNetDecoder(nn.Module):
    def __init__(self, in_channels=1280, num_classes=21):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)
        return x


class EfficientNet(ModelBase):
    def __init__(
        self,
        num_classes=21,
        file_path: str = "./model_saves/efficientnet.pth",
        device=None,
        use_wandb=False,
    ):
        super().__init__(file_path=file_path, device=device, use_wandb=use_wandb)
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone = backbone.features.to(self.device)
        self.output_dim = 1280
        self.decoder = UNetDecoder(
            in_channels=self.output_dim, num_classes=num_classes
        ).to(self.device)

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.backbone(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


def compute_loss_effnet(pred, target):
    if isinstance(target, list):
        target = torch.stack(target)

    target = target.long().to(pred.device)

    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(
            pred, size=target.shape[-2:], mode="bilinear", align_corners=False
        )

    return F.cross_entropy(pred, target, ignore_index=255)
