import torch
import torch.nn as nn
import torchvision.models as models
from model_base import BaseModel  # Adjust path if needed


class Model1(ModelBase):
    def __init__(
        self,
        input_dim,
        output_dim,
        file_path: str = "./model_saves/model1.pth",
        device=None,
    ):
        """
        Initializes the model with the specified input and output dimensions, file path for saving the model,
        and the device to be used for computation.
        Args:
            input_dim (int): The dimensionality of the input features.
            output_dim (int): The dimensionality of the output features.
            file_path (str, optional): The file path where the model will be saved. Defaults to "./model_saves/model1.pth".
            device (torch.device, optional): The device to be used for computation (e.g., 'cpu' or 'cuda'). Defaults to None.
        """
        super().__init__(file_path, device)

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * (input_dim // 4) * (input_dim // 4), 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
