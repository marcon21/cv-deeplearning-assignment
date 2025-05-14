import numpy as np
import torch
import data
import torch.optim as optim
import torch.nn as nn
import argparse
import wandb
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from models.unet import UNet
from models.model1 import Model1
from models.swin import SwinTransformer, compute_loss_swin, get_scheduler
# from models.model3 import Model3

if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="Train segmentation models.")
    parser.add_argument(
        "--epochs", type=int, nargs="+", default=[10],
        help="List of training epochs for each model (default: [10]). If one value is provided, it will be used for all models."
    )
    parser.add_argument(
        "--batch_sizes", type=int, nargs="+", default=[4, 4, 4], help="List of three batch sizes (default: [4, 4, 4])."
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
        "--decoder",
        type=str,
        default="simple",
        choices=["simple", "deeplab", "aspp"],
        help="Backbone model size; options: tiny, base, small (default: tiny)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of workers for data loading (default: 4). If one value is provided, it will be used for all models."
    )
    parser.add_argument(
        "--learning_rates",
        type=float,
        nargs="+",
        default=[6e-5, 6e-4],
        help="Learning rate for the optimizer (default: 0.0001). If one value is provided, it will be used for all models."
    )
    parser.add_argument(
        "--weight_decays",
        type=float,
        nargs="+",
        default=[[1e-4, 5e-4]],
        help="Weight decay for the optimizer (default: 0.01). If one value is provided, it will be used for all models."
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
    else:
        if len(args.batch_sizes) != len(args.models):
            raise ValueError("Batch sizes must either be one value or match the number of models.")
        batch_sizes = args.batch_sizes

    if len(args.epochs) == 1:
        epochs_list = [args.epochs[0]] * len(args.models)
    elif len(args.epochs) != len(args.models):
        raise ValueError("If multiple epochs are provided, they must match the number of models.")
    else:
        epochs_list = [int(epoch) for epoch in args.epochs]


    # Set backbone for Swin if needed
    backbone = None
    if "Swin" in args.models:
        backbone_map = {
            "tiny": "microsoft/swin-tiny-patch4-window7-224",
            "base": "microsoft/swin-base-patch4-window7-224",
            "small": "microsoft/swin-small-patch4-window7-224",
            "base384": "microsoft/swin-base-patch4-window7-384",
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
        "Swin": lambda: SwinTransformer(
            num_classes=21,
            decoder=str(args.decoder),
            model_name=backbone,
            device=device,
            file_path=f"./model_saves",
            use_wandb=True,
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

    lr = 0.0001
    
    # Execute training for each run
    for i, run_params in enumerate(runs):
        model = run_params["constructor"]()

        train_loader, test_loader, eval_loader = data.load_data(
            train=0.80,
            test=0.10,
            eval=0.10,
            root_dir="VOC",
            batch_size=run_params["batch_size"],
            num_workers=args.workers,
            grayscale=False,
            resize= (224, 224) if model.model_name == "Swin" else (256, 256),
        )

        print(f"Training {model.model_name}...")
        if model.model_name == "SwinTransformer":
            lr1 = args.learning_rates[0]
            lr2 = args.learning_rates[1]
            weight_decay = args.weight_decays[0]
            weight_decay2 = args.weight_decays[1]
            if args.backbone == "tiny":
                optimizer = optim.AdamW([
                        {"params": model.backbone.parameters(), "lr": lr1, "weight_decay": weight_decay},
                        {"params": model.decoder.parameters(), "lr": lr2, "weight_decay": weight_decay2},
                    ])
            elif args.backbone == "base":
                optimizer = optim.AdamW([
                        {"params": model.backbone.parameters(), "lr": lr1, "weight_decay": weight_decay},
                        {"params": model.decoder.parameters(), "lr": lr2, "weight_decay": weight_decay2},
                    ])
            elif args.backbone == "small":
                optimizer = optim.AdamW([
                        {"params": model.backbone.parameters(), "lr": lr1, "weight_decay": weight_decay},
                        {"params": model.decoder.parameters(), "lr": lr2, "weight_decay": weight_decay2},
                    ])
            scheduler = get_scheduler(optimizer, warmup_steps=int(run_params["epochs"] * len(train_loader)*0.05), total_steps=run_params["epochs"] * len(train_loader))
            print(f"Using optimizer: {optimizer}, scheduler: {scheduler}, learning rates: {lr1}, {lr2}")
            

        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)
        if model.model_name == "Swin":
            loss_fn = compute_loss_swin
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        model.to(device)
        model.train_model(
            train_loader=train_loader,
            test_loader=test_loader,
            eval_loader=eval_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=run_params["epochs"],
            scheduler=scheduler if model.model_name == "Swin" else None,
        )
        model.save()
