# cv-deeplearning-assignment

### Model Implementation

All the models are implemented in the `models` directory. All the models inherit from the class `BaseModel` from `models/base_model.py` all the training methods and logging functions.

The data loading is done in the file `data.py`.

To train the model run `train.py`

`inference.py` is still TODO
for now can be run with

```
python inference.py --model_name <model_name> --model_path <model_path> --image_path <image_path>

python inference.py --model_class UNet --model_path ./model_saves/unet.pth
```
