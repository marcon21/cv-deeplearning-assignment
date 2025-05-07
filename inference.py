import os
import torch
import data
from PIL import Image
import numpy as np

# Dynamically import model class
import importlib


def get_model_class(class_name):
    if class_name == "UNet":
        from models.unet import UNet

        return UNet
    elif class_name == "Model1":
        from models.model1 import Model1

        return Model1
    # Add more models as needed
    else:
        raise ValueError(f"Unknown model class: {class_name}")


def save_image(tensor, path, is_mask=False):
    array = tensor.cpu().numpy()
    if is_mask:
        # VOC masks: single channel, map 255 to 0 for visualization
        array = np.where(array == 255, 0, array)
        array = array.astype(np.uint8)
        img = Image.fromarray(array)
    else:
        # Input: (C, H, W) -> (H, W, C), unnormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        array = array.transpose(1, 2, 0)
        array = array * std + mean
        array = np.clip(array * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(array)
    img.save(path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_class", type=str, required=True, help="Model class name (e.g., UNet)"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model weights"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of examples to run inference on",
    )
    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Load data (use test set)
    _, test_loader, _ = data.load_data(batch_size=1)

    # Dynamically get model class
    ModelClass = get_model_class(args.model_class)
    if args.model_class == "UNet":
        model = ModelClass(input_channels=3, output_channels=21, device=device)
    elif args.model_class == "Model1":
        # Example: Model1(input_height, input_width, output_dim, ...)
        model = ModelClass(
            input_height=256, input_width=256, output_dim=21, device=device
        )
    else:
        raise ValueError("Unknown model class")
    model.load(args.model_path)
    model.to(device)
    model.eval()

    # Prepare output directory
    out_dir = os.path.join("out", "examples")
    os.makedirs(out_dir, exist_ok=True)

    # Run inference on num_examples
    count = 0
    for images, masks in test_loader:
        if count >= args.num_examples:
            break
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
        # Save input, ground truth, prediction
        save_image(images[0], os.path.join(out_dir, f"{count}_input.png"))
        save_image(masks[0], os.path.join(out_dir, f"{count}_gt.png"), is_mask=True)
        save_image(preds[0], os.path.join(out_dir, f"{count}_pred.png"), is_mask=True)
        count += 1
    print(f"Saved {count} examples to {out_dir}")


if __name__ == "__main__":
    main()
