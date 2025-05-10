import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinModel
from .model_base import ModelBase

class Model2(ModelBase):
    def __init__(self, num_classes=21, decoder=None, model_name="microsoft/swin-tiny-patch4-window7-224", file_path=None, device=None, use_wandb=True):
        super().__init__(file_path, device, use_wandb)
        self.backbone = SwinModel.from_pretrained(model_name).to(device)
        self.output_dim = self.backbone.config.hidden_size
        self.decoder = nn.Sequential(
            nn.Conv2d(self.output_dim, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1)
        ).to(device)

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.backbone(pixel_values=x).last_hidden_state  # (B, 49, 768)
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.decoder(x)  # (B, num_classes, H, W)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x  # (B, num_classes, H, W)
    
def compute_loss_swin(pred, target):
    # Resize logits to match target size
    pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
    loss = F.cross_entropy(pred, target)
    return loss