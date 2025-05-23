import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import SwinModel
from .model_base import ModelBase
from torch.optim.lr_scheduler import LambdaLR
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import math
from functools import partial
from .aspp_decoder import ASPPDecoder


def compute_loss_swin(pred, target):
    pred = F.interpolate(
        pred, size=target.shape[-2:], mode="bilinear", align_corners=False
    ).contiguous()
    loss = F.cross_entropy(pred, target, ignore_index=255)
    return loss


def make_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))  # avoids 0
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return lr_lambda


def get_scheduler(optimizer, warmup_steps, total_steps):
    lr_lambda = make_lr_lambda(warmup_steps, total_steps)
    return LambdaLR(optimizer, lr_lambda)


class SwinTransformer(ModelBase):
    def __init__(
        self,
        num_classes=21,
        model_name="microsoft/swin-tiny-patch4-window7-224",
        file_path=None,
        device=None,
        use_wandb=True,
        decoder=None,
    ):
        super().__init__(file_path, device, use_wandb)
        self.backbone = SwinModel.from_pretrained(model_name).to(device)
        self.output_dim = self.backbone.config.hidden_size
        self.decoder_name = decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(self.output_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )
        if decoder == "deeplab":
            self.decoder = DeepLabHead(
                in_channels=self.output_dim, num_classes=num_classes
            )
        elif decoder == "aspp":
            self.decoder = ASPPDecoder(
                in_channels=self.output_dim, num_classes=num_classes
            )

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.backbone(pixel_values=x).last_hidden_state
        B, N, C = x.shape
        H = W = int(N**0.5)
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x

    @staticmethod
    def _compute_miou(pred, target, num_classes=21, ignore_index=255):
        if pred.dim() == 4:
            pred = torch.argmax(pred, dim=1)
        assert pred.shape == target.shape, "Shape mismatch between pred and target"

        valid_mask = target != ignore_index
        pred = pred[valid_mask]
        target = target[valid_mask]
        ious = []
        for cls in range(num_classes):
            if cls == ignore_index:
                continue
            pred_mask = pred == cls
            target_mask = target == cls
            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()
            if union == 0:
                continue
            iou = intersection / union
            ious.append(iou)

        if not ious:
            return 0.0
        return sum(ious) / len(ious)
