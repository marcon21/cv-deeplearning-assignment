import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_base import ModelBase
from models.aspp_decoder import ASPPDecoder
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNet(ModelBase):
    def __init__(self, num_classes=21, file_path: str = "./model_saves/efficientnet.pth", device=None):
        super().__init__(file_path=file_path, device=device)
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).features.to(device)
        self.output_dim = 1280  
        self.decoder = ASPPDecoder(in_channels=self.output_dim, num_classes=num_classes).to(device)

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.backbone(x) 
        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x

def compute_loss_effnet( pred, target):
    if isinstance(target, list):
        target = torch.stack(target)

    target = target.long().to(pred.device)

    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)

    return F.cross_entropy(pred, target, ignore_index=255)
