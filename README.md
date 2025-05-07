# cv-deeplearning-assignment

### Model Implementation

All the models are implemented in the `models` directory. All the models inherit from the class `BaseModel` from `models/base_model.py` all the training methods and logging functions.

The data loading is done in the file `data.py`.

## Training

To train the model, run `train.py` with customizable options:

```
python train.py [--epochs N] [--batch_size N] [--models MODEL1 MODEL2 ...] [--device DEVICE]
```

### Options

- `--epochs N` : Number of training epochs (default: 10)
- `--batch_size N` : Batch size (default: 16)
- `--models ...` : List of model class names to train (e.g., UNet Model1). If not specified, no models are trained by default. Example: `--models UNet Model1`
- `--device DEVICE` : Device to use for training. Options: `auto`, `cpu`, `cuda`, `mps` (default: `auto`).
  - `auto` will select `cuda` if available, then `mps`, then `cpu`.

#### Example usage

Train UNet for 20 epochs on CUDA (if available):

```
python train.py --epochs 20 --models UNet --device cuda
```

Train both UNet and Model1 with batch size 8 on CPU:

```
python train.py --batch_size 8 --models UNet Model1 --device cpu
```

## Inference

`inference.py` is still TODO
for now can be run with

```
python inference.py --model_name <model_name> --model_path <model_path> --image_path <image_path>

python inference.py --model_class UNet --model_path ./model_saves/unet.pth
```
