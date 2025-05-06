import torch
import torch.nn as nn
import torchvision.models as models
from models.model_base import BaseModel 


class EfficientNetSegmentation(BaseModel):
    def __init__(self, num_classes=21):
        super().__init__()
        self.num_classes = num_classes

        # Load EfficientNet-B0 as encoder
        backbone = models.efficientnet_b0(pretrained=True)
        self.encoder = backbone.features  # outputs: [B, 1280, H/32, W/32]

        # Decoder: Upsample 5x to get back to input resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, kernel_size=2, stride=2),  # H/16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),   # H/8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),   # H/4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),    # H/2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)  # H
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
