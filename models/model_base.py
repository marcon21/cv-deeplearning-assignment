from torch import nn
from torch.nn import functional as F
import torch
from typing import Union
from sklearn.metrics import f1_score
import wandb
from tqdm import tqdm


class ModelBase(nn.Module):
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
        self.test_history = []

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

        self.model_name = self.__class__.__name__

        # Initialize wandb
        wandb.init(
            project="cv-deeplearning-assignment",
            name=self.model_name,
            config={
                "model_name": self.model_name,
            },
        )

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
        train_loader,  # DataLoader for training data
        test_loader=None,
        eval_loader=None,
        epochs: int = 1,
        optimizer: torch.optim.Optimizer = None,
        loss_fn: callable = None,
    ) -> None:
        """
        Trains the model for a specified number of epochs using the provided data loaders.
        Args:
            train_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader, optional): DataLoader for validation data. Defaults to None.
            eval_loader (DataLoader, optional): DataLoader for evaluation data. Defaults to None.
            epochs (int, optional): Number of training epochs. Defaults to 1.
            optimizer (torch.optim.Optimizer, optional): Optimizer to use for training. If None, Adam optimizer is used. Defaults to None.
            loss_fn (callable, optional): Loss function to use. If None, CrossEntropyLoss is used. Defaults to None.
        Returns:
            None
        """

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        self.train()

        for epoch in range(epochs):
            epoch_loss = 0.0

            for inputs, targets in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False
            ):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                targets = (
                    targets.long()
                )  # Ensure targets are of type Long for CrossEntropyLoss

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            self.train_history.append(avg_loss)
            wandb.log({f"{self.model_name}/train_loss": avg_loss, "epoch": epoch})
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}", end="")

            if test_loader is not None:
                test_loss, test_accuracy, f1 = self.evaluate_model(test_loader, loss_fn)
                wandb.log(
                    {
                        f"{self.model_name}/test_loss": test_loss,
                        f"{self.model_name}/test_accuracy": test_accuracy,
                        f"{self.model_name}/f1_score": f1,
                        "epoch": epoch,
                    }
                )
                self.test_history.append(test_loss)
                print(f" | Test Loss: {test_loss:.4f}", end="")

            print()

        if eval_loader is not None:
            eval_loss, eval_accuracy, f1 = self.evaluate_model(eval_loader, loss_fn)
            wandb.log(
                {
                    f"{self.model_name}/eval_loss": eval_loss,
                    f"{self.model_name}/eval_accuracy": eval_accuracy,
                    f"{self.model_name}/eval_f1_score": f1,
                    "epoch": epochs - 1,
                }
            )
            self.eval_history.append(eval_loss)
            print(
                f"Eval Loss: {eval_loss:.4f}",
                f" | Eval Accuracy: {eval_accuracy:.4f}",
                f" | Eval F1 Score: {f1:.4f}",
            )

        self.eval()
        print("Training complete.")

    def evaluate_model(
        self, data_loader, loss_fn: callable = None
    ) -> tuple[float, float, float]:
        """
        Evaluates the model on the provided data loader.
        Args:
            data_loader (DataLoader): DataLoader for evaluation data.
            loss_fn (callable, optional): Loss function to use. If None, CrossEntropyLoss is used. Defaults to None.

        Returns:
            float: The average loss over the evaluation dataset.
            float: The accuracy of the model on the evaluation dataset.
            float: The F1 score of the model on the evaluation dataset.
        """

        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = targets.long()
                outputs = self(inputs)
                loss = loss_fn(outputs, targets)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.numel()

                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(targets.cpu().numpy().flatten())

        accuracy = correct / total if total > 0 else 0
        f1 = f1_score(all_labels, all_preds, average="macro") if total > 0 else 0

        avg_loss = total_loss / len(data_loader)
        return avg_loss, accuracy, f1
