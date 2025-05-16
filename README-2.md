# cv-deeplearning-assignment

### Inference and Visualization

The `inference.py` script not only performs inference on test examples but also provides additional visualization options.

**Key Features:**

- Loads test data using `data.load_data`.
- Dynamically imports model classes (e.g., `UNet`, `EfficientNet`, `Swin`) based on the `--model_class` argument.
- Loads pre-trained model weights specified by `--model_paths`.
- Runs inference on a specified number of test images.
- Outputs for each example:
  - Original input image (after normalization).
  - Ground truth mask (with a Pascal VOC color palette for clarity).
  - The prediction mask (using the same palette).
  - A concatenated image that joins the input, ground truth, and prediction images side-by-side.
  - An overlapped image where:
    - Ground truth areas are blended in red.
    - Predicted regions are blended in green.

### Command-Line Arguments for `inference.py`

- `--model_class` (required): One or more model class names (e.g., `UNet EfficientNet Swin`).
- `--model_paths` (required): Corresponding file paths for the model weights for each class specified. Must be in the same order as `--model_class`.
- `--num_examples` : Number of examples on which to run inference (default: 5).

The device for inference (CUDA, MPS, CPU) is selected automatically. For `SwinTransformer`, specific configurations (e.g., backbone size, decoder type) are currently defined within `inference.py` during model instantiation and are not set via command-line arguments in this script.

### Example Usage for `inference.py`

To run inference using a pre-trained `Swin` model and a `UNet` model, on 10 examples:

```bash
python inference.py --model_class Swin UNet --model_paths ./model_saves/swin_model.pth ./model_saves/unet_model.pth --num_examples 10
```

You can also include `EfficientNet` if you have a trained model for it.

The generated images are saved in the `out/examples/<model_name>/` directory for each model.

### Function Overview (`inference.py`)

- `get_model_class(class_name)`: Dynamically imports and returns the requested model class.
- `get_voc_palette(num_classes=21)`: Returns the Pascal VOC color palette adjusted to 256×3 entries.
- `save_image(tensor, path, is_mask=False)`: Saves a tensor as an image; applies appropriate scaling and palette for masks.
- `concat_and_save_images(...)`: Concatenates the input image, ground truth, and prediction for easy comparison.
- `save_overlapped_images(...)`: Overlays ground truth (red) and prediction (green) on the input image using blending.

---

### Training Models

The `train.py` script supports training multiple models, including `UNet`, `EfficientNet`, and `SwinTransformer`. It allows customization of training parameters.

### Command-Line Arguments for `train.py`

- `--models` (required): List of model class names to train (e.g., `UNet Swin EfficientNet`).
- `--epochs`: List of training epochs for each model (default: `[10]`). If one value is provided, it's used for all models.
- `--batch_sizes`: List of batch sizes for each model (default: `[4]`). If one value is provided, it's used for all models.
- `--device`: Device to use (`auto`, `cpu`, `cuda`, `mps`; default: `auto`).
- `--load_weights`: List of paths to model weights to load before training (in order of models specified in `--models`). Use 'None' or skip for models without pre-trained weights.
- `--workers`: Number of workers for data loading (default: `0`).
- `--learning_rates`: List of learning rates (default: `[6e-5, 6e-4]`).
  - If training `Swin` or `EfficientNet`: The first two values from this list are used as `[backbone_lr, decoder_lr]`.
  - If training other models (e.g., `UNet`): The first value in this list (`learning_rates[0]`) is used as its learning rate.
  - **Caution**: If training multiple models simultaneously (e.g., `Swin` and `UNet`), be aware that `Swin`/`EfficientNet` (for their backbone LR) and `UNet` will all attempt to use `learning_rates[0]`. The script does not currently consume LRs sequentially for different model types from this list. Plan your learning rate list and model order accordingly.
- `--weight_decays`: List of weight decay values (default: `[1e-4, 5e-4]`).
  - If training `Swin` or `EfficientNet`: The first two values from this list are used as `[backbone_wd, decoder_wd]`.
  - For other models (e.g., `UNet`): This argument is _not_ used by `train.py` to set weight decay; their optimizers (e.g., Adam) will use their own default weight decay (typically 0 for Adam).
  - **Caution**: Similar to learning rates, `Swin`/`EfficientNet` will use `weight_decays[0]` and `weight_decays[1]`.

**Swin Transformer and EfficientNet Related Arguments (used if "Swin" or "EfficientNet" is in `--models`):**

- `--backbone`: Backbone model size.
  - For `SwinTransformer` (`tiny`, `base`, `small`; default: `tiny`), this defines the architecture.
  - For `EfficientNet`, this argument is used by `train.py` to construct the filename for saving/loading weights (e.g., `efficientnet_tiny.pth`) but does not alter the `EfficientNet` architecture itself (which is typically a fixed variant like B0, B1, etc., defined within the model code).
- `--decoder`: Decoder type for `SwinTransformer` (`simple`, `deeplab`, `aspp`; default: `simple`). This argument is not used by `EfficientNet`.

### Example Usage for `train.py`

To train a `SwinTransformer` model with a `tiny` backbone and `aspp` decoder for 20 epochs with batch size 8, and a `UNet` model for 15 epochs with batch size 16:

```bash
python train.py --models Swin UNet --epochs 20 15 --batch_sizes 8 16 --backbone tiny --decoder aspp --learning_rates 5e-5 5e-4 1e-3 --weight_decays 1e-4 1e-4 0
```

In this example:

- `Swin` uses backbone LR `5e-5` (from `learning_rates[0]`), decoder LR `5e-4` (from `learning_rates[1]`). It uses backbone WD `1e-4` (from `weight_decays[0]`) and decoder WD `1e-4` (from `weight_decays[1]`).
- `UNet` uses LR `5e-5` (from `learning_rates[0]`, due to current script logic). The third learning rate `1e-3` is not automatically assigned to `UNet` in this mixed setup. The weight decay for `UNet` is determined by its Adam optimizer's default (0), as the `0` in `--weight_decays ... 0` is not passed to the `UNet` optimizer by `train.py`.
