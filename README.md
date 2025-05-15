# cv-deeplearning-assignment

### Inference and Visualization

The `inference.py` script not only performs inference on test examples but also provides additional visualization options. The script:

• Loads test data using `data.load_data` and selects a subset of examples.  
• Dynamically imports the model class (e.g., `UNet`, `Model1`, or `Swin`) based on the provided `--model_class` argument.  
• Loads model weights via the `--model_path` argument and runs inference on the test images.

The script outputs several files for each example:
• The original input image (after normalization).  
• The ground truth mask (with a Pascal VOC color palette for clarity).  
• The prediction mask (using the same palette).  
• A concatenated image that joins the input, ground truth, and prediction images side-by-side.  
• An overlapped image where:
  - Ground truth areas are blended in red.
  - Predicted regions are blended in green.

### Command-Line Arguments

- `--model_classes` : List of model class name (e.g., `UNet`, `Model1`, `Swin`).
- `--model_paths`  : File paths for the model weights af all modelss
- `--num_examples` : Number of examples on which to run inference (default: 5).

The device for inference is selected automatically (prioritizing CUDA, then MPS, then CPU).

### Example Usage

To run inference on a `Swin` model with a `tiny` backbone and `simple` decoder for 5 examples:

  python inference.py --model_class Swin --model_path ./model_saves/swin.pth --num_examples 5 --backbone tiny --decoder simple

The generated images are saved in the folder `out/examples`.

### Function Overview

- get_model_class(class_name): Dynamically imports and returns the requested model class.
- get_voc_palette(num_classes=21): Returns the Pascal VOC color palette adjusted to 256×3 entries.
- save_image(tensor, path, is_mask=False): Saves a tensor as an image; applies appropriate scaling and palette for masks.
- concat_and_save_images(...): Concatenates the input image, ground truth, and prediction for easy comparison.
- save_overlapped_images(...): Overlays ground truth (red) and prediction (green) on the input image using blending.

This enhanced inference script streamlines evaluation by combining prediction, visual comparison, and result saving in one run.

### Training

The `train.py` script supports training multiple models, including `UNet`, `Model1`, and `Swin`. It allows customization of training parameters such as epochs, batch sizes, learning rates, and weight decays.

### Command-Line Arguments

- `--models` : List of model class names to train (e.g., `UNet`, `Model1`, `Swin`).
- `--epochs` : List of training epochs for each model (default: `[10]`).
- `--batch_sizes` : List of batch sizes for each model (default: `[4, 4, 4]`).
- `--backbone` : Backbone model size for `Swin` (options: `tiny`, `base`, `small`; default: `tiny`).
- `--decoder` : Decoder type for `Swin` (options: `simple`, `deeplab`; default: `simple`).
- `--learning_rates` : Learning rates for the optimizer (default: `[[6e-5, 6e-4]]`).
- `--weight_decays` : Weight decays for the optimizer (default: `[[1e-4, 5e-4]]`).

### Example Usage

To train a `Swin` model with a `tiny` backbone and `simple` decoder:

  python train.py --models Swin --epochs 20 --batch_sizes 8 --backbone tiny --decoder simple
