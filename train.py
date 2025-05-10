import numpy as np
import torch
import data
import torch.optim as optim
import torch.nn as nn
import argparse

from models.unet import UNet
from models.model1 import Model1
from models.swin import Model2, compute_loss_swin
# from models.model3 import Model3

if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="Train segmentation models.")
    parser.add_argument(
        "--epochs", type=int, nargs="+", default=[10],
        help="List of training epochs for each model (default: [10]). If one value is provided, it will be used for all models."
    )
    parser.add_argument(
        "--batch_sizes", type=int, nargs=3, default=[4], help="List of three batch sizes (default: [4]). If one value is provided, it will be used for all models."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[],
        help="List of model class names to train.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use: auto, cpu, cuda, or mps (default: auto)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="tiny",
        choices=["tiny", "base", "small"],
        help="Backbone model size; options: tiny, base, small (default: tiny)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=[8],
        help="Number of workers for data loading (default: 4). If one value is provided, it will be used for all models."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=[0.001],
        help="Learning rate for the optimizer (default: 0.001). If one value is provided, it will be used for all models."
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

    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    # Process batch sizes and epochs into lists for each run
    if len(args.batch_sizes) == 1:
        batch_sizes = [args.batch_sizes[0]] * len(args.models)
    elif len(args.batch_sizes) != len(args.models):
        raise ValueError("Batch sizes must either be one value or match the number of models.")
    else:
        batch_sizes = args.batch_sizes

    if len(args.epochs) == 1:
        epochs_list = [args.epochs[0]] * len(args.models)
    elif len(args.epochs) != len(args.models):
        raise ValueError("If multiple epochs are provided, they must match the number of models.")
    else:
        epochs_list = [int(epoch) for epoch in args.epochs]


    # Set backbone for Model2 if needed
    backbone = None
    if "Model2" in args.models:
        backbone_map = {
            "tiny": "microsoft/swin-tiny-patch4-window7-224",
            "base": "microsoft/swin-base-patch4-window7-224",
            "small": "microsoft/swin-small-patch4-window7-224",
        }
        try:
            backbone = backbone_map[args.backbone]
        except KeyError:
            raise ValueError(f"Unknown backbone: {args.backbone}. Choose from {list(backbone_map.keys())}.")
        print(f"Using backbone: {backbone}")

    # Map model class names to constructors with required args
    model_classes = {
        "UNet": lambda: UNet(input_channels=3, output_channels=21, device=device),
        "Model1": lambda: Model1(input_height=256, input_width=256, output_dim=21, device=device),
        "Model2": lambda: Model2(
            num_classes=21,
            decoder=None,
            model_name=backbone,
            device=device,
            file_path=f"./model_saves/model2_{args.backbone}.pth",
            use_wandb=False,
        ),
    }

    # Build parameter sets for each run
    runs = []
    for model_name, batch_size, epochs in zip(args.models, batch_sizes, epochs_list):
        if model_name not in model_classes:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_classes.keys())}.")
        runs.append({
            "constructor": model_classes[model_name],
            "batch_size": batch_size,
            "epochs": epochs
        })

    # Set learning rate from the command-line arguments
    lr = args.learning_rate[0]
    
    # Execute training for each run
    for run_params in runs:
        model = run_params["constructor"]()

        train_loader, test_loader, eval_loader = data.load_data(
            train=0.80,
            test=0.10,
            eval=0.10,
            root_dir="./VOC",
            batch_size=run_params["batch_size"],
            num_workers=args.workers,
            grayscale=False,
        )

        print(f"Training {model.model_name}...")
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        if model.model_name == "Model2":
            loss_fn = compute_loss_swin
        model.to(device)
        model.train_model(
            train_loader=train_loader,
            test_loader=test_loader,
            eval_loader=eval_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=run_params["epochs"],
        )
        model.save()
