import torch
from models.model_base import ModelBase
from torch import nn


class UNet(ModelBase):
    def __init__(
        self,
        input_channels,
        output_channels,
        file_path: str = "./model_saves/unet.pth",
        device=None,
        use_wandb: bool = True,
        dropout_rate: float = 0.3,
    ):
        """
        Initializes the U-Net model.
        Args:
            input_channels (int): Number of channels in the input image.
            output_channels (int): Number of channels in the output segmentation map.
            file_path (str, optional): The file path where the model will be saved. Defaults to "./model_saves/unet.pth".
            device (torch.device, optional): The device to be used for computation. Defaults to None.
            use_wandb (bool, optional): Whether to use wandb for logging. Defaults to True.
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.3.
        """
        super().__init__(file_path, device, use_wandb=use_wandb)
        self.dropout_rate = dropout_rate

        # Encoder (Downsampling path)
        self.enc_conv1 = self.double_conv(input_channels, 64, use_dropout=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_conv2 = self.double_conv(64, 128, use_dropout=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_conv3 = self.double_conv(128, 256, use_dropout=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_conv4 = self.double_conv(256, 512, use_dropout=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck_conv = self.double_conv(512, 1024, use_dropout=True)

        # Decoder (Upsampling path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_conv4 = self.double_conv(1024, 512, use_dropout=False)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3 = self.double_conv(512, 256, use_dropout=False)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = self.double_conv(256, 128, use_dropout=False)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = self.double_conv(128, 64, use_dropout=False)

        # Output layer
        self.out_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels, use_dropout=False):
        """Helper function for a block of two 3x3 convolutions with ReLU activation and optional dropout."""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(self.dropout_rate))
        return nn.Sequential(*layers)

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
