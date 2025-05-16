# cv-deeplearning-assignment

## Overview

This repository provides scripts for training and evaluating deep learning models for semantic segmentation on the Pascal VOC dataset. You can train models like UNet, EfficientNet, and Swin Transformer, and run inference to visualize predictions.

---

## 1. Training Models (`train.py`)

The `train.py` script allows you to train one or more segmentation models with customizable settings.

### Usage

```bash
python train.py --models <MODEL_NAMES> [options]
```

### Required Arguments

- `--models`: List of model class names to train. Supported: `UNet`, `EfficientNet`, `Swin`.

### Common Options

- `--epochs`: List of epochs for each model (default: `[10]`).
- `--batch_sizes`: List of batch sizes for each model (default: `[4]`).
- `--device`: Device to use (`auto`, `cpu`, `cuda`, `mps`; default: `auto`).
- `--learning_rates`: Learning rates. For `Swin`/`EfficientNet`, provide two values: `[backbone_lr, decoder_lr]`. For others, only the first value is used.
- `--weight_decays`: Weight decay values. For `Swin`/`EfficientNet`, provide two values: `[backbone_wd, decoder_wd]`.
- `--backbone`: (For Swin) Backbone size: `tiny`, `base`, or `small` (default: `tiny`).
- `--decoder`: (For Swin) Decoder type: `simple`, `deeplab`, or `aspp` (default: `simple`).
- `--load_weights`: (Optional) List of paths to pre-trained weights for each model.
- `--workers`: Number of data loader workers (default: `0`).

### Example: Train Swin and UNet

```bash
python train.py --models Swin UNet --epochs 20 15 --batch_sizes 8 16 \
  --backbone tiny --decoder aspp \
  --learning_rates 5e-5 5e-4 1e-3 --weight_decays 1e-4 1e-4 0
```

- This trains a Swin Transformer (tiny, aspp decoder) for 20 epochs (batch 8) and a UNet for 15 epochs (batch 16).
- The first two learning rates/weight decays are used for Swin; the first is used for UNet.

**Note:**

- The order of arguments matters. The number of epochs, batch sizes, and weights should match the number of models (or provide a single value to use for all).
- For `Swin` and `EfficientNet`, always provide two learning rates and two weight decays.

---

## 2. Inference and Visualization (`inference.py`)

The `inference.py` script runs inference using trained models and visualizes the results. It can process images from the validation set or a custom folder.

### Usage

```bash
python inference.py --model_class <MODEL_NAMES> --model_paths <WEIGHT_PATHS> [options]
```

### Required Arguments

- `--model_class`: One or more model class names (e.g., `UNet EfficientNet Swin`).
- `--model_paths`: Corresponding paths to model weights (same order as `--model_class`).

### Options

- `--num_examples`: Number of examples to run inference on.
  - **If using `--input_folder` and not specified, all images in the folder will be processed.**
  - **If not using `--input_folder`, defaults to 5.**
- `--input_folder`: (Optional) Path to a folder of images to run inference on. If not provided, uses the validation set.

### Example: Inference with Swin and UNet

```bash
python inference.py --model_class Swin UNet \
  --model_paths ./model_saves/swin_model.pth ./model_saves/unet_model.pth \
  --num_examples 10
```

- This runs inference on 10 validation images using both models.

### Example: Inference on Custom Images

```bash
python inference.py --model_class UNet \
  --model_paths ./model_saves/unet_model.pth \
  --input_folder ./my_images
```

- This runs inference on **all images** from `./my_images` using UNet (unless you specify `--num_examples`).

### Output

- Results are saved in `out/examples/<model_name>/` for each model.
- For each example, you get:
  - The input image
  - (If available) Ground truth mask
  - The predicted mask
  - A concatenated image (input | ground truth | prediction)
  - An overlapped image (input with ground truth in red, prediction in green)

### Notes

- The order of `--model_class` and `--model_paths` must match.
- Device is selected automatically (CUDA, MPS, or CPU).
- For Swin, backbone/decoder are set in the script (not via command-line).
- **If you use `--input_folder` and do not specify `--num_examples`, all images in the folder will be processed.**
- **If you do not use `--input_folder`, the default is 5 examples unless you specify otherwise.**

---

## Troubleshooting

- If you get errors about argument counts, check that the number of models, weights, epochs, and batch sizes match or are set to a single value.
- Make sure your model weights exist and are compatible with the model class.
- For custom images, ensure they are in a supported format (`.jpg`, `.jpeg`, `.png`, `.bmp`).

---

## Quick Start

1. **Train a model:**
   ```bash
   python train.py --models UNet --epochs 10 --batch_sizes 4
   ```
2. **Run inference:**
   ```bash
   python inference.py --model_class UNet --model_paths ./model_saves/unet_model.pth --num_examples 5
   ```

---

For more details, see the code and comments in `train.py` and `inference.py`.
