import os
import torch
import data
from PIL import Image
import numpy as np
from tqdm import tqdm


def get_model_class(class_name):
    if class_name == "UNet":
        from models.unet import UNet

        return UNet
    elif class_name == "EfficientNet":
        from models.efficientnet import EfficientNet

        return EfficientNet

    elif class_name == "Swin":
        from models.swin import SwinTransformer

        return SwinTransformer
    else:
        raise ValueError(f"Unknown model class: {class_name}")


def get_voc_palette(num_classes=21):
    # Pascal VOC color palette
    palette = [
        0,
        0,
        0,
        128,
        0,
        0,
        0,
        128,
        0,
        128,
        128,
        0,
        0,
        0,
        128,
        128,
        0,
        128,
        0,
        128,
        128,
        128,
        128,
        128,
        64,
        0,
        0,
        192,
        0,
        0,
        64,
        128,
        0,
        192,
        128,
        0,
        64,
        0,
        128,
        192,
        0,
        128,
        64,
        128,
        128,
        192,
        128,
        128,
        0,
        64,
        0,
        128,
        64,
        0,
        0,
        192,
        0,
        128,
        192,
        0,
        0,
        64,
        128,
    ]
    # Pad palette to 256*3
    palette += [0] * (256 * 3 - len(palette))
    return palette


def tensor_to_pil(tensor, is_mask=False):
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
    return img


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


def concat_and_save_images_from_pil(input_img, gt_img, pred_img, out_path):
    imgs = [input_img, gt_img, pred_img]
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


def save_overlapped_images(input_path, gt_path, pred_path, out_path):
    input_img = Image.open(input_path).convert("RGBA")
    gt_img = Image.open(gt_path).convert("L")
    pred_img = Image.open(pred_path).convert("L")

    overlapped_img = input_img.copy()
    pixels = overlapped_img.load()
    input_pixels = input_img.load()
    alpha_gt = 0.5
    alpha_pred = 0.5

    def blend(background, overlay, alpha):
        return tuple(
            int(alpha * overlay[i] + (1 - alpha) * background[i]) for i in range(3)
        ) + (255,)

    for x in range(input_img.width):
        for y in range(input_img.height):
            bg_pixel = input_pixels[x, y]
            gt_val = gt_img.getpixel((x, y))
            pred_val = pred_img.getpixel((x, y))
            if gt_val != 0:  # ground truth: red overlay
                pixels[x, y] = blend(bg_pixel, (255, 0, 0), alpha_gt)
            elif pred_val != 0:  # prediction: green overlay
                pixels[x, y] = blend(bg_pixel, (0, 255, 0), alpha_pred)
            else:
                pixels[x, y] = bg_pixel

    overlapped_img = overlapped_img.convert("RGB")
    overlapped_img.save(out_path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_class",
        type=str,
        required=True,
        help="Model classes, e.g., UNet, EfficientNet, Swin",
        default=["UNet", "EfficientNet", "Swin"],
        nargs="+",
    )
    parser.add_argument(
        "--model_paths",
        type=str,
        required=True,
        help="Paths to model weights",
        nargs="+",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Number of examples to run inference on (default: 5 for validation set, all for input_folder if not specified)",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        help="Path to a folder of images to run inference on (optional)",
        default=None,
    )
    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Helper for preprocessing single images
    from torchvision import transforms

    preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # If input_folder is provided, use images from there
    if args.input_folder is not None:
        image_paths = [
            os.path.join(args.input_folder, fname)
            for fname in os.listdir(args.input_folder)
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        image_paths = sorted(image_paths)
        if args.num_examples is not None:
            image_paths = image_paths[: args.num_examples]
        eval_loader = None  # Not used
    else:
        # Load data (use validation set)
        num_examples = args.num_examples if args.num_examples is not None else 5
        _, _, eval_loader = data.load_data(batch_size=1, size=(256, 256))

    # Dynamically get model class
    for model_name, index in zip(args.model_class, range(len(args.model_class))):
        if model_name not in ["UNet", "EfficientNet", "Swin"]:
            raise ValueError(f"Unknown model class: {model_name}")
        ModelClass = get_model_class(model_name)
        model_instance = None
        if model_name == "UNet":
            model_instance = ModelClass(
                input_channels=3,
                output_channels=21,
                device=device,
                use_wandb=False,
                file_path=None,
            )
        elif model_name == "EfficientNet":
            model_instance = ModelClass(
                num_classes=21,
                file_path=None,
                device=device,
                use_wandb=False,
            )
        elif model_name == "Swin":
            model_instance = ModelClass(
                num_classes=21,
                decoder="aspp",
                model_name="microsoft/swin-small-patch4-window7-224",
                file_path=None,
                device=device,
                use_wandb=False,
            )

        model_instance.load(args.model_paths[index])
        model_instance.to(device)
        model_instance.eval()

        out_dir = os.path.join(
            "out", "examples", model_name
        )  # Modified to include model_name in out_dir
        os.makedirs(out_dir, exist_ok=True)

        count = 0
        if args.input_folder is not None:
            image_iter = tqdm(
                image_paths,
                total=len(image_paths),
                desc=f"{model_name} inference (folder)",
                leave=True,
            )
            for img_path in image_iter:
                if args.num_examples is not None and count >= args.num_examples:
                    break
                img = Image.open(img_path).convert("RGB")
                input_tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model_instance(input_tensor)
                    preds = torch.argmax(outputs, dim=1)
                input_pil = tensor_to_pil(input_tensor[0])
                pred_pil = tensor_to_pil(preds[0], is_mask=True)
                concat_path = os.path.join(out_dir, f"{count}_concat.png")
                # Only input and prediction (no GT mask)
                concat_and_save_images_from_pil(
                    input_pil, pred_pil, pred_pil, concat_path
                )
                count += 1
                image_iter.set_postfix({"Saved": count})
            print(f"Saved {count} examples for {model_name} to {out_dir}")
        else:
            image_iter = tqdm(
                eval_loader,
                total=num_examples,
                desc=f"{model_name} inference",
                leave=True,
            )
            for images, masks in image_iter:
                if count >= num_examples:
                    break
                images = images.to(device)
                with torch.no_grad():
                    outputs = model_instance(images)
                    preds = torch.argmax(outputs, dim=1)

                input_pil = tensor_to_pil(images[0])
                gt_pil = tensor_to_pil(masks[0], is_mask=True)
                pred_pil = tensor_to_pil(preds[0], is_mask=True)

                concat_path = os.path.join(out_dir, f"{count}_concat.png")
                concat_and_save_images_from_pil(
                    input_pil, gt_pil, pred_pil, concat_path
                )

                count += 1
                image_iter.set_postfix({"Saved": count})
            print(f"Saved {count} examples for {model_name} to {out_dir}")


if __name__ == "__main__":
    main()
