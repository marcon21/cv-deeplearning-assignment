# cv-deeplearning-assignment

### Inference and Visualization

The `inference.py` script not only performs inference on test examples but also provides additional visualization options.

**Key Features:**

- Loads test data using `data.load_data`.
- Dynamically imports model classes (e.g., `UNet`, `Model1`, `Swin`) based on the `--model_class` argument.
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

- `--model_class` (required): One or more model class names (e.g., `UNet Model1 Swin`).
- `--model_paths` (required): Corresponding file paths for the model weights for each class specified. Must be in the same order as `--model_class`.
- `--num_examples` : Number of examples on which to run inference (default: 5).

The device for inference (CUDA, MPS, CPU) is selected automatically. For `SwinTransformer`, specific configurations (e.g., backbone size, decoder type) are currently defined within `inference.py` during model instantiation and are not set via command-line arguments in this script.

### Example Usage for `inference.py`

To run inference using a pre-trained `Swin` model and a `UNet` model, on 10 examples:

```bash
python inference.py --model_class Swin UNet --model_paths ./model_saves/swin_model.pth ./model_saves/unet_model.pth --num_examples 10
```

The generated images are saved in the `out/examples/<model_name>/` directory for each model.

### Function Overview (`inference.py`)

- `get_model_class(class_name)`: Dynamically imports and returns the requested model class.
- `get_voc_palette(num_classes=21)`: Returns the Pascal VOC color palette adjusted to 256×3 entries.
- `save_image(tensor, path, is_mask=False)`: Saves a tensor as an image; applies appropriate scaling and palette for masks.
- `concat_and_save_images(...)`: Concatenates the input image, ground truth, and prediction for easy comparison.
- `save_overlapped_images(...)`: Overlays ground truth (red) and prediction (green) on the input image using blending.

---

### Training Models

The `train.py` script supports training multiple models, including `UNet`, `Model1`, and `SwinTransformer`. It allows customization of training parameters.

### Command-Line Arguments for `train.py`

- `--models` (required): List of model class names to train (e.g., `UNet Swin Model1`).
- `--epochs`: List of training epochs for each model (default: `[10]`). If one value is provided, it's used for all models.
- `--batch_sizes`: List of batch sizes for each model (default: `[4]`). If one value is provided, it's used for all models.
- `--device`: Device to use (`auto`, `cpu`, `cuda`, `mps`; default: `auto`).
- `--load_weights`: List of paths to model weights to load before training (in order of models specified in `--models`). Use 'None' or skip for models without pre-trained weights.
- `--workers`: Number of workers for data loading (default: `0`).
- `--learning_rates`: List of learning rates (default: `[6e-5, 6e-4]`). For `Swin`, expects two values `[backbone_lr, decoder_lr]`. For other models, the first value in the list is used.
- `--weight_decays`: List of weight decay values (default: `[1e-4, 5e-4]`). For `Swin`, expects two values `[backbone_wd, decoder_wd]`. For other models, this argument might not be directly used or configured internally by their optimizers.

**Swin Transformer Specific Arguments (used if "Swin" is in `--models`):**

- `--backbone`: Backbone model size (`tiny`, `base`, `small`; default: `tiny`).
- `--decoder`: Decoder type (`simple`, `deeplab`, `aspp`; default: `simple`).

### Example Usage for `train.py`

To train a `SwinTransformer` model with a `tiny` backbone and `aspp` decoder for 20 epochs with batch size 8, and a `UNet` model for 15 epochs with batch size 16:

```bash
python train.py --models Swin UNet --epochs 20 15 --batch_sizes 8 16 --backbone tiny --decoder aspp --learning_rates 5e-5 5e-4 1e-3 --weight_decays 1e-4 1e-4 0
```

In this example:

- `Swin` uses backbone LR 5e-5, decoder LR 5e-4, backbone WD 1e-4, and decoder WD 1e-4.
- `UNet` uses LR 1e-3. The weight decay for `UNet` would depend on its internal Adam optimizer setup if not explicitly passed or if it uses the first value from `--weight_decays` (0 in this case).
