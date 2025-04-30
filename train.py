import numpy as np
import torch
import data
import torch.optim as optim
import torch.nn as nn

from models.model1 import Model1

# from models.model2 import Model2
# from models.model3 import Model3

if __name__ == "__main__":
    import os

    os.environ["WANDB_SILENT"] = "true"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    X, y, X_test, y_test, X_eval, y_eval = data.load_data(
        train=0.80, test=0.10, eval=0.10
    )

    models = [
        Model1(input_dim=10, output_dim=1, device=device),
        # Model2(device=device),
        # Model3(device=device),
    ]

    for model in models:
        print(f"Training {model.model_name}...")
        print(model)

        lr = 0.001
        batch_size = 32
        epochs = 10
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        model.to(device)

        model.train_model(
            X=X,
            y=y,
            X_test=X_test,
            y_test=y_test,
            X_eval=X_eval,
            y_eval=y_eval,
            optimizer=optimizer,
            loss_fn=loss_fn,
            batch_size=batch_size,
            epochs=epochs,
        )

        model.save()

    # model.plot_train_history()
