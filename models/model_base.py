from torch import nn
from torch.nn import functional as F
import torch
from typing import Union
from sklearn.metrics import f1_score
import wandb


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
        X: torch.Tensor,
        y: torch.Tensor,
        X_test: torch.Tensor = None,
        y_test: torch.Tensor = None,
        X_eval: torch.Tensor = None,
        y_eval: torch.Tensor = None,
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
            y (torch.Tensor, optional): Target data tensor.
            X_test (torch.Tensor, optional): Validation input data tensor. Defaults to None.
            y_test (torch.Tensor, optional): Validation target data tensor. Defaults to None.
            X_eval (torch.Tensor, optional): Evaluation input data tensor. Defaults to None.
            y_eval (torch.Tensor, optional): Evaluation target data tensor. Defaults to None.
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
            self.train()
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()

            self.train_history.append(loss.item())
            # Replace TensorBoard logging with wandb
            wandb.log({f"{self.model_name}/train_loss": loss.item(), "epoch": epoch})

            # Terminal output for training loss
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4f}", end="")

            if X_test is not None and y_test is not None:
                val_loss, val_accuracy, f1 = self.evaluate_model(
                    X_test, y_test, batch_size, loss_fn
                )
                wandb.log(
                    {
                        f"{self.model_name}/val_loss": val_loss,
                        f"{self.model_name}/val_accuracy": val_accuracy,
                        f"{self.model_name}/f1_score": f1,
                        "epoch": epoch,
                    }
                )
                self.test_history.append(val_loss)
                # Terminal output for validation
                print(
                    f" | Val Loss: {val_loss:.4f}",
                )

            print()  # Newline after each epoch

        if X_eval is not None and y_eval is not None:
            eval_loss, eval_accuracy, f1 = self.evaluate_model(
                X_eval, y_eval, batch_size, loss_fn
            )
            wandb.log(
                {
                    f"{self.model_name}/eval_loss": eval_loss,
                    f"{self.model_name}/eval_accuracy": eval_accuracy,
                    f"{self.model_name}/eval_f1_score": f1,
                    "epoch": epoch,
                }
            )
            self.eval_history.append(eval_loss)
            # Terminal output for evaluation
            print(
                f"Eval Loss: {eval_loss:.4f}",
                f" | Eval Accuracy: {eval_accuracy:.4f}",
                f" | Eval F1 Score: {f1:.4f}",
            )

        self.eval()
        print("Training complete.")

    def evaluate_model(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 32,
        loss_fn: callable = None,
    ) -> tuple[float, float, float]:
        """
        Evaluates the model on the provided data.
        Args:
            X (torch.Tensor): Input data tensor.
            y (torch.Tensor, optional): Target data tensor. Defaults to None.
            batch_size (int, optional): Size of each evaluation batch. Defaults to 32.
            loss_fn (callable, optional): Loss function to use. If None, MSELoss is used. Defaults to None.
        Returns:
            float: The average loss over the evaluation dataset.
            float: The accuracy of the model on the evaluation dataset.
            float: The F1 score of the model on the evaluation dataset.
        """

        if loss_fn is None:
            loss_fn = nn.MSELoss()

        self.eval()
        y = y.to(self.device)
        X = X.to(self.device)
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                outputs = self(batch_X)
                loss = loss_fn(outputs, batch_y)
                total_loss += loss.item()

                # Assuming binary classification for accuracy and F1 score
                preds = torch.round(torch.sigmoid(outputs))
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        accuracy = correct / total
        f1 = f1_score(all_labels, all_preds, average="binary")
        avg_loss = total_loss / len(dataloader)
        return avg_loss, accuracy, f1
