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


def get_voc_palette(num_classes=21):
    # Pascal VOC color palette
    palette = [
        0,
        0,
        0,  # 0=background
        128,
        0,
        0,  # 1=aeroplane
        0,
        128,
        0,  # 2=bicycle
        128,
        128,
        0,  # 3=bird
        0,
        0,
        128,  # 4=boat
        128,
        0,
        128,  # 5=bottle
        0,
        128,
        128,  # 6=bus
        128,
        128,
        128,  # 7=car
        64,
        0,
        0,  # 8=cat
        192,
        0,
        0,  # 9=chair
        64,
        128,
        0,  # 10=cow
        192,
        128,
        0,  # 11=diningtable
        64,
        0,
        128,  # 12=dog
        192,
        0,
        128,  # 13=horse
        64,
        128,
        128,  # 14=motorbike
        192,
        128,
        128,  # 15=person
        0,
        64,
        0,  # 16=potted plant
        128,
        64,
        0,  # 17=sheep
        0,
        192,
        0,  # 18=sofa
        128,
        192,
        0,  # 19=train
        0,
        64,
        128,  # 20=tv/monitor
    ]
    # Pad palette to 256*3
    palette += [0] * (256 * 3 - len(palette))
    return palette


def save_image(tensor, path, is_mask=False):
    array = tensor.cpu().numpy()
    if is_mask:
        # VOC masks: single channel, map 255 to 0 for visualization
        array = np.where(array == 255, 0, array)
        array = array.astype(np.uint8)
        img = Image.fromarray(array, mode="P")
        img.putpalette(get_voc_palette())
    else:
        # Input: (C, H, W) -> (H, W, C), unnormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        array = array.transpose(1, 2, 0)
        array = array * std + mean
        array = np.clip(array * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(array)
    img.save(path)


def concat_and_save_images(input_path, gt_path, pred_path, out_path):
    imgs = [Image.open(p) for p in [input_path, gt_path, pred_path]]
    # Ensure all images are RGB for consistency
    imgs = [img.convert("RGB") for img in imgs]
    widths, heights = zip(*(img.size for img in imgs))
    total_width = sum(widths)
    max_height = max(heights)
    new_img = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for img in imgs:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width
    new_img.save(out_path)


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
    _, test_loader, _ = data.load_data(batch_size=1, size=(256, 256))

    # Dynamically get model class
    ModelClass = get_model_class(args.model_class)
    if args.model_class == "UNet":
        model = ModelClass(
            input_channels=3, output_channels=21, device=device, use_wandb=False
        )
    elif args.model_class == "Model1":
        # Example: Model1(input_height, input_width, output_dim, ...)
        model = ModelClass(
            input_height=256,
            input_width=256,
            output_dim=21,
            device=device,
            use_wandb=False,
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
        input_path = os.path.join(out_dir, f"{count}_input.png")
        gt_path = os.path.join(out_dir, f"{count}_gt.png")
        pred_path = os.path.join(out_dir, f"{count}_pred.png")
        save_image(images[0], input_path)
        save_image(masks[0], gt_path, is_mask=True)
        save_image(preds[0], pred_path, is_mask=True)
        # Concatenate and save
        concat_path = os.path.join(out_dir, f"{count}_concat.png")
        concat_and_save_images(input_path, gt_path, pred_path, concat_path)
        count += 1
    print(f"Saved {count} examples to {out_dir}")


if __name__ == "__main__":
    main()
