import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from models.model_base import ModelBase


class EfficientNet(ModelBase):
    def __init__(self, num_classes=21, file_path: str = "./model_saves/efficientnet.pth", device=None):
        super().__init__(file_path=file_path, device=device)
        self.num_classes = num_classes

        # Encoder: EfficientNet-B0 with pretrained weights
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.encoder = backbone.features  # [B, 1280, H/32, W/32]

        # Decoder: Upsample back to original resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x

    def compute_loss(self, pred, target):
        # If target is a list of masks
        if isinstance(target, list):
            target = torch.stack(target)

        target = target.long().to(pred.device)

        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)

        return F.cross_entropy(pred, target, ignore_index=255)
