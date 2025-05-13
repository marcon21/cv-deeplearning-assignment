import torch
from models.model_base import ModelBase
from torch import nn


class UNet(ModelBase):
    def __init__(
        self,
        input_channels,  # Changed from input_dim
        output_channels,  # Changed from output_dim
        file_path: str = "./model_saves/unet.pth",
        device=None,
        use_wandb: bool = True,
    ):
        """
        Initializes the U-Net model.
        Args:
            input_channels (int): Number of channels in the input image.
            output_channels (int): Number of channels in the output segmentation map.
            file_path (str, optional): The file path where the model will be saved. Defaults to "./model_saves/unet.pth".
            device (torch.device, optional): The device to be used for computation. Defaults to None.
            use_wandb (bool, optional): Whether to use wandb for logging. Defaults to True.
        """
        super().__init__(file_path, device, use_wandb=use_wandb)

        # Encoder (Downsampling path)
        self.enc_conv1 = self.double_conv(input_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_conv2 = self.double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_conv3 = self.double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_conv4 = self.double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck_conv = self.double_conv(512, 1024)

        # Decoder (Upsampling path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_conv4 = self.double_conv(
            1024, 512
        )  # 512 (from upconv) + 512 (from enc_conv4)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3 = self.double_conv(
            512, 256
        )  # 256 (from upconv) + 256 (from enc_conv3)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = self.double_conv(
            256, 128
        )  # 128 (from upconv) + 128 (from enc_conv2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = self.double_conv(
            128, 64
        )  # 64 (from upconv) + 64 (from enc_conv1)

        # Output layer
        self.out_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        """Helper function for a block of two 3x3 convolutions with ReLU activation."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        skip1 = self.enc_conv1(x)
        x = self.pool1(skip1)
        skip2 = self.enc_conv2(x)
        x = self.pool2(skip2)
        skip3 = self.enc_conv3(x)
        x = self.pool3(skip3)
        skip4 = self.enc_conv4(x)
        x = self.pool4(skip4)

        # Bottleneck
        x = self.bottleneck_conv(x)

        # Decoder
        x = self.upconv4(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.dec_conv4(x)

        x = self.upconv3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.dec_conv3(x)

        x = self.upconv2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec_conv2(x)

        x = self.upconv1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec_conv1(x)

        # Output
        x = self.out_conv(x)
        return x