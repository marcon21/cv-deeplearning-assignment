import numpy as np
import torch
from model1 import Model1

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Set the random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    model = Model1(input_dim=10, output_dim=1, device=device)
    print(model)

    # model.train_model()
    # model.save()
    # model.plot_train_history()
