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
        "--epochs",
        type=int,
        nargs="+",
        default=[10],
        help="List of training epochs for each model (default: [10]). If one value is provided, it will be used for all models.",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[4],
        help="List of batch sizes (default: [4]). If one value is provided, it will be used for all models. Otherwise, must match the number of models.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[],
        help="List of model class names to train (e.g., UNet Swin Model1).",
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
        help="List of paths to model weights to load before training (in order of models specified in --models). Use 'None' or skip for models without pre-trained weights.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="tiny",
        choices=["tiny", "base", "small"],
        help="Backbone model size for Swin Transformer; options: tiny, base, small (default: tiny)",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="simple",
        choices=["simple", "deeplab", "aspp"],
        help="Decoder type for Swin Transformer; options: simple, deeplab, aspp (default: simple)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of workers for data loading (default: 0).",
    )
    parser.add_argument(
        "--learning_rates",
        type=float,
        nargs="+",
        default=[6e-5, 6e-4],
        help="List of learning rates (default: [6e-5, 6e-4]). For Swin, expects two values [backbone_lr, decoder_lr]. For other models, the first value is used.",
    )
    parser.add_argument(
        "--weight_decays",
        type=float,
        nargs="+",
        default=[1e-4, 5e-4],
        help="List of weight decay values (default: [1e-4, 5e-4]). For Swin, expects two values [backbone_wd, decoder_wd]. Other models typically configure this within their optimizer or don't use it from this arg.",
    )
    args = parser.parse_args()

    os.environ["WANDB_SILENT"] = "true"

    if not args.models:
        print("No models specified. Exiting.")
        exit()

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

    num_models = len(args.models)
    if len(args.batch_sizes) == 1:
        batch_sizes = [args.batch_sizes[0]] * num_models
    elif len(args.batch_sizes) != num_models:
        raise ValueError(
            "Batch sizes must either be one value or match the number of models."
        )
    else:
        batch_sizes = args.batch_sizes

    if len(args.epochs) == 1:
        epochs_list = [args.epochs[0]] * num_models
    elif len(args.epochs) != num_models:
        raise ValueError(
            "If multiple epochs are provided, they must match the number of models."
        )
    else:
        epochs_list = [int(epoch) for epoch in args.epochs]

    weights_to_load = args.load_weights
    if weights_to_load is None:
        weights_to_load = [None] * num_models
    elif len(weights_to_load) != num_models:
        if len(weights_to_load) < num_models:
            print(
                f"Warning: Fewer weight paths ({len(weights_to_load)}) provided than models ({num_models}). Remaining models will not have weights loaded."
            )
            weights_to_load.extend([None] * (num_models - len(weights_to_load)))
        else:  # len(weights_to_load) > num_models
            raise ValueError(
                "Cannot provide more weight paths than the number of models."
            )

    swin_backbone_name = None
    if "Swin" in args.models:
        backbone_map = {
            "tiny": "microsoft/swin-tiny-patch4-window7-224",
            "base": "microsoft/swin-base-patch4-window7-224",
            "small": "microsoft/swin-small-patch4-window7-224",
        }
        try:
            swin_backbone_name = backbone_map[args.backbone]
        except KeyError:
            raise ValueError(
                f"Unknown Swin backbone: {args.backbone}. Choose from {list(backbone_map.keys())}."
            )
        print(
            f"Selected Swin backbone configuration: {swin_backbone_name} (will be used if a Swin model is trained)"
        )

    model_classes = {
        "UNet": lambda: UNet(input_channels=3, output_channels=21, device=device),
        "Model1": lambda: Model1(
            input_height=256, input_width=256, output_dim=21, device=device
        ),
        "Swin": lambda: SwinTransformer(
            num_classes=21,
            decoder=str(args.decoder),
            model_name=swin_backbone_name,
            device=device,
            file_path=f"./model_saves",
            use_wandb=True,
        ),
    }

    for i, model_key_name in enumerate(args.models):
        current_batch_size = batch_sizes[i]
        current_epochs = epochs_list[i]
        current_weight_path = weights_to_load[i]

        print(f"\n--- Training Model {i+1}/{num_models}: {model_key_name} ---")
        print(f"Batch Size: {current_batch_size}, Epochs: {current_epochs}")
        if current_weight_path and current_weight_path.lower() != "none":
            print(f"Attempting to load weights from: {current_weight_path}")

        if model_key_name not in model_classes:
            raise ValueError(
                f"Unknown model: {model_key_name}. Choose from {list(model_classes.keys())}."
            )
        model = model_classes[model_key_name]()

        if current_weight_path and current_weight_path.lower() != "none":
            try:
                model.load(current_weight_path)
                print(
                    f"Successfully loaded weights for {model_key_name} from {current_weight_path}"
                )
            except Exception as e:
                print(
                    f"Error loading weights for {model_key_name} from {current_weight_path}: {e}"
                )

        resize_dim = (224, 224) if model_key_name == "Swin" else (256, 256)
        print(f"Using resize dimensions: {resize_dim} for {model_key_name}")

        train_loader, test_loader, eval_loader = data.load_data(
            train=0.80,
            test=0.10,
            eval=0.10,
            root_dir="VOC",
            batch_size=current_batch_size,
            num_workers=args.workers,
            grayscale=False,
            resize=resize_dim,
        )

        scheduler = None

        if model_key_name == "Swin":
            if not swin_backbone_name:
                raise ValueError(
                    "Swin model specified but backbone name not resolved. This is an internal error."
                )
            if len(args.learning_rates) < 2:
                raise ValueError(
                    "Swin Transformer requires at least two learning rates (for backbone and decoder). Provide using --learning_rates lr_backbone lr_decoder."
                )
            lr1 = args.learning_rates[0]
            lr2 = args.learning_rates[1]

            if len(args.weight_decays) < 2:
                raise ValueError(
                    "Swin Transformer requires at least two weight decay values (for backbone and decoder). Provide using --weight_decays wd_backbone wd_decoder."
                )
            wd1 = args.weight_decays[0]
            wd2 = args.weight_decays[1]

            if not hasattr(model, "backbone") or not hasattr(model, "decoder"):
                raise AttributeError(
                    f"Swin model {model_key_name} does not have 'backbone' or 'decoder' attributes required for optimizer setup."
                )

            optimizer = optim.AdamW(
                [
                    {
                        "params": model.backbone.parameters(),
                        "lr": lr1,
                        "weight_decay": wd1,
                    },
                    {
                        "params": model.decoder.parameters(),
                        "lr": lr2,
                        "weight_decay": wd2,
                    },
                ]
            )
            scheduler = get_scheduler(
                optimizer,
                warmup_steps=int(current_epochs * len(train_loader) * 0.05),
                total_steps=current_epochs * len(train_loader),
            )
            print(
                f"Optimizer for Swin: AdamW. LR_backbone: {lr1}, LR_decoder: {lr2}. WD_backbone: {wd1}, WD_decoder: {wd2}. Scheduler enabled."
            )
        else:
            if not args.learning_rates:
                raise ValueError("Learning rates not specified.")
            current_lr = args.learning_rates[0]
            optimizer = optim.Adam(model.parameters(), lr=current_lr)
            print(f"Optimizer for {model_key_name}: Adam. Learning Rate: {current_lr}.")

        if model_key_name == "Swin":
            loss_fn = compute_loss_swin
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        print(
            f"Using loss function: {loss_fn.__class__.__name__ if isinstance(loss_fn, nn.Module) else loss_fn.__name__}"
        )

        model.to(device)
        display_model_name = getattr(model, "model_name", model_key_name)
        print(
            f"Starting training for {display_model_name} on {device} for {current_epochs} epochs..."
        )

        if not hasattr(model, "train_model"):
            raise AttributeError(
                f"Model {model_key_name} does not have a 'train_model' method."
            )

        model.train_model(
            train_loader=train_loader,
            test_loader=test_loader,
            eval_loader=eval_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=current_epochs,
            scheduler=scheduler,
        )

        if not hasattr(model, "save"):
            print(
                f"Warning: Model {model_key_name} does not have a 'save' method. Cannot save model."
            )
        else:
            print(f"Saving model {display_model_name}...")
            model.save()

    print("\nAll specified models have been trained.")
