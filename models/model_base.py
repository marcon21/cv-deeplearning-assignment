import os
from torch import nn
from torch.nn import functional as F
import torch
from typing import Union, Tuple
from sklearn.metrics import f1_score
import wandb
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Subset, DataLoader


class ModelBase(nn.Module):
    def __init__(
        self,
        file_path: str = None,
        device: Union[str, torch.device] = None,
        use_wandb: bool = True,
    ):
        """
        Initializes the model instance.
        Args:
            file_path: Path to a directory for saving model data. Defaults to 'model_saves'.
            device: The device to run the model on ('cuda', 'mps', or 'cpu').
                If not specified, automatically selects 'cuda' if available, then 'mps', then 'cpu'.
            use_wandb: Whether to use wandb for logging. Defaults to True.
        Attributes:
            file_path: Stores the provided file path for saving models.
            train_history: Keeps track of training loss.
            eval_history: Keeps track of evaluation loss.
            test_history: Keeps track of test loss.
            device: The device selected for computation.
            use_wandb: Whether to use wandb for logging.
        """

        super().__init__()
        self.file_path = file_path
        self.train_history = []
        self.eval_history = []
        self.test_history = []
        self.use_wandb = use_wandb

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
        if file_path is None:
            default_dir = os.path.dirname(os.path.abspath(__file__))
            self.file_path = os.path.join(os.path.dirname(default_dir), "model_saves")
        else:
            self.file_path = file_path
        os.makedirs(self.file_path, exist_ok=True)

        if self.use_wandb:
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
            file_path: The path where the model state will be saved.
                If None, uses a default name within the instance's `file_path` directory.
        """

        if file_path is None:
            # Default to saving in the designated model directory with model name
            file_path = os.path.join(self.file_path, f"{self.model_name}_default.pth")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        try:
            torch.save(self.state_dict(), file_path)
        except Exception as e:
            print(f"Error saving model to {file_path}: {e}")
            fallback_path = os.path.join(self.file_path, "fallback_model.pth")
            try:
                torch.save(self.state_dict(), fallback_path)
                print(f"Model saved to fallback location: {fallback_path}")
            except Exception as e:
                print(f"Error saving model to fallback location {fallback_path}: {e}")
                print("Model saving failed. Check your file path and permissions.")

    def load(self, file_path: str = None):
        """
        Loads the model's state dictionary from a file and sets the model to evaluation mode.
        Args:
            file_path: Path to the file containing the saved state dictionary.
                If None, attempts to load from a default path in `self.file_path`.
        """

        if file_path is None:
            file_path = os.path.join(self.file_path, f"{self.model_name}_default.pth")

        state_dict = torch.load(file_path, map_location=self.device, weights_only=True)
        self.load_state_dict(state_dict)
        self.eval()

    def plot_train_history(self):
        """Plots the training loss history of the model."""
        import matplotlib.pyplot as plt

        plt.plot(self.train_history, label="Train Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training History")
        plt.legend()
        plt.show()

    def get_subset_loader(self, data_loader, subset_size):
        """
        Returns a DataLoader for a random subset of the given data_loader's dataset.
        Args:
            data_loader: The original DataLoader.
            subset_size: The desired size of the subset.
        """
        dataset = data_loader.dataset
        total_size = len(dataset)
        indices = np.random.choice(
            total_size, min(subset_size, total_size), replace=False
        )
        return DataLoader(
            Subset(dataset, indices),
            batch_size=data_loader.batch_size,
            shuffle=False,
            num_workers=getattr(data_loader, "num_workers", 0),
            collate_fn=getattr(data_loader, "collate_fn", None),
        )

    def train_model(
        self,
        train_loader,
        test_loader=None,
        eval_loader=None,
        epochs: int = 1,
        optimizer: torch.optim.Optimizer = None,
        loss_fn: callable = None,
        scheduler: callable = None,
    ) -> None:
        """
        Trains the model for a specified number of epochs.
        Args:
            train_loader: DataLoader for training data.
            test_loader: DataLoader for validation data (optional).
            eval_loader: DataLoader for evaluation data (optional).
            epochs: Number of training epochs.
            optimizer: Optimizer to use. If None, Adam optimizer is used.
            loss_fn: Loss function to use. If None, CrossEntropyLoss is used.
            scheduler: Learning rate scheduler (optional).
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
                targets = targets.long()

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                    wandb.log(
                        {
                            f"{self.model_name}/lr": scheduler.get_last_lr()[0],
                            "epoch": epoch,
                        }
                    )

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            self.train_history.append(avg_loss)
            if self.use_wandb:
                wandb.log({f"{self.model_name}/train_loss": avg_loss, "epoch": epoch})
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}", end="")

            # Validation logic
            if test_loader is not None:
                test_loss, test_accuracy, f1, miou = self.evaluate_model(
                    test_loader, loss_fn
                )
                if self.use_wandb:
                    wandb.log(
                        {
                            f"{self.model_name}/test_loss": test_loss,
                            f"{self.model_name}/test_accuracy": test_accuracy,
                            f"{self.model_name}/f1_score": f1,
                            f"{self.model_name}/miou": miou,
                            "epoch": epoch,
                        }
                    )
                self.test_history.append(test_loss)
                print(f" | Test Loss: {test_loss:.4f}", end="")

            print()  # Newline after epoch status

            # Save model every epoch
            epoch_save_path = os.path.join(
                self.file_path, f"{self.model_name}_epoch_{epoch+1}.pth"
            )
            self.save(epoch_save_path)

            FREQUENCY_SAVE = 10
            # Save a specific checkpoint every FREQUENCY_SAVE epochs
            if (epoch + 1) % FREQUENCY_SAVE == 0:
                checkpoint_save_path = os.path.join(
                    self.file_path, f"{self.model_name}_checkpoint_epoch_{epoch+1}.pth"
                )
                self.save(checkpoint_save_path)
                print(
                    f"Saved checkpoint model to {checkpoint_save_path} at epoch {epoch+1}"
                )

        if eval_loader is not None:
            eval_loss, eval_accuracy, f1, miou = self.evaluate_model(
                eval_loader, loss_fn
            )
            if self.use_wandb:
                wandb.log(
                    {
                        f"{self.model_name}/eval_loss": eval_loss,
                        f"{self.model_name}/eval_accuracy": eval_accuracy,
                        f"{self.model_name}/eval_f1_score": f1,
                        f"{self.model_name}/eval_miou": miou,
                        "epoch": epochs - 1,
                    }
                )
            self.eval_history.append(eval_loss)
            print(
                f"Eval Loss: {eval_loss:.4f}",
                f" | Eval Accuracy: {eval_accuracy:.4f}",
                f" | Eval F1 Score: {f1:.4f}",
                f" | Eval mIoU: {miou:.4f}",
            )

        self.eval()
        print("Training complete.")

    @staticmethod
    def _compute_miou(pred, target, num_classes=21):
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        intersection = torch.zeros(num_classes).to(pred.device)
        union = torch.zeros(num_classes).to(pred.device)

        for i in range(num_classes):
            intersection[i] = ((pred == i) & (target == i)).sum()
            union[i] = ((pred == i) | (target == i)).sum()

        miou = (intersection / (union + 1e-6)).mean().item()
        return miou

    def evaluate_model(
        self, data_loader, loss_fn: callable = None
    ) -> Tuple[float, float, float, float]:
        """
        Evaluates the model on the provided data loader.
        Args:
            data_loader: DataLoader for evaluation data.
            loss_fn: Loss function to use. If None, CrossEntropyLoss is used.

        Returns:
            Average loss, accuracy, F1 score, and mIoU.
        """

        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_miou = []

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

                miou = self._compute_miou(preds, targets)
                all_miou.append(miou)

        accuracy = correct / total if total > 0 else 0
        f1 = f1_score(all_labels, all_preds, average="macro") if total > 0 else 0
        miou = sum(all_miou) / len(all_miou) if all_miou else 0

        avg_loss = total_loss / len(data_loader)
        return avg_loss, accuracy, f1, miou
