from torch import nn
from torch.nn import functional as F
import torch
from typing import Union


class ModelTemplate(nn.Module):
    def __init__(self, file_path: str = None, device: Union[str, torch.device] = None):
        """
        Initializes the model instance.
        Args:
            file_path (str, optional): Path to a file for loading or saving model data. Defaults to None.
            device (str or torch.device, optional): The device to run the model on ('cuda', 'mps', or 'cpu').
                If not specified, automatically selects 'cuda' if available, otherwise 'mps', otherwise 'cpu'.
        Attributes:
            file_path (str or None): Stores the provided file path.
            train_history (list): Keeps track of training history.
            eval_history (list): Keeps track of evaluation history.
            device (str): The device selected for computation.
        """

        super().__init__()
        self.file_path = file_path
        self.train_history = []
        self.eval_history = []

        self.device = (
            device
            if device
            else (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )

        self.to(self.device)

    def save(self, file_path: str = None):
        """
        Saves the model's state dictionary to the specified file path.
        Args:
            file_path (str, optional): The path where the model state will be saved.
                If None, uses the instance's default file_path attribute.
        Returns:
            None
        """

        if file_path is None:
            file_path = self.file_path

        torch.save(self.state_dict(), file_path)

    def load(self, file_path: str = None):
        """
        Loads the model's state dictionary from a file and sets the model to evaluation mode.
        Args:
            file_path (str, optional): Path to the file containing the saved state dictionary.
                If None, uses the default file path stored in self.file_path.
        Returns:
            None
        """

        if file_path is None:
            file_path = self.file_path

        self.load_state_dict(torch.load(file_path))
        self.eval()

    def plot_train_history(self):
        """
        Plots the training history of the model.
        Returns:
            None
        """
        import matplotlib.pyplot as plt

        plt.plot(self.train_history, label="Train Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training History")
        plt.legend()
        plt.show()

    def train_model(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 1,
        batch_size: int = 32,
        optimizer: torch.optim.Optimizer = None,
        lr: float = 0.001,
        loss_fn: callable = None,
    ) -> None:
        """
        Trains the model for a specified number of epochs using the provided data.
        Args:
            X (torch.Tensor): Input data tensor.
            y (torch.Tensor, optional): Target data tensor. Defaults to None.
            epochs (int, optional): Number of training epochs. Defaults to 1.
            batch_size (int, optional): Size of each training batch. Defaults to 32.
            optimizer (torch.optim.Optimizer, optional): Optimizer to use for training. If None, Adam optimizer is used. Defaults to None.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            loss_fn (callable, optional): Loss function to use. If None, MSELoss is used. Defaults to None.
        Returns:
            None
        """

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        self.train()

        y = y.to(self.device)
        X = X.to(self.device)
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()

            self.train_history.append(loss.item())
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        self.eval()
        print("Training complete.")
