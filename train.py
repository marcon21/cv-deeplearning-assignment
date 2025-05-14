import numpy as np
import torch
import data
import torch.optim as optim
import torch.nn as nn
import argparse

from models.unet import UNet
from models.model1 import Model1

# from models.model2 import Model2
# from models.model3 import Model3

if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="Train segmentation models.")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[],
        help="List of model class names to train (default: UNet)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use: auto, cpu, cuda, or mps (default: auto)",
    )
    parser.add_argument(
        "--load_weights",
        type=str,
        nargs="*",
        default=None,
        help="List of paths to model weights to load before training (in order of models).",
    )
    args = parser.parse_args()

    os.environ["WANDB_SILENT"] = "true"

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # device = "cpu"

    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    batch_size = args.batch_size
    epochs = args.epochs
    lr = 0.001

    train_loader, test_loader, eval_loader = data.load_data(
        train=0.80, test=0.10, eval=0.10, batch_size=batch_size
    )

    # Map model class names to constructors and their required args
    model_classes = {
        "UNet": lambda: UNet(input_channels=3, output_channels=21, device=device),
        "Model1": lambda: Model1(
            input_height=256, input_width=256, output_dim=21, device=device
        ),
    }

    models = []
    for name in args.models:
        if name in model_classes:
            models.append(model_classes[name]())
        else:
            raise ValueError(f"Unknown model class: {name}")

    # Load weights if provided
    if args.load_weights is not None:
        if len(args.load_weights) > len(models):
            raise ValueError("More weight paths provided than models.")
        for i, weight_path in enumerate(args.load_weights):
            if weight_path and weight_path.lower() != "none":
                print(f"Loading weights for {models[i].model_name} from {weight_path}")
                models[i].load(weight_path)

    for model in models:
        print(f"Training {model.model_name}...")
        # print(model)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)

        model.to(device)

        model.train_model(
            train_loader=train_loader,
            test_loader=test_loader,
            eval_loader=eval_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=epochs,
        )

        model.save()

    # model.plot_train_history()
