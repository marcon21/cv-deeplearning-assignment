import numpy as np
import torch
import data
import torch.optim as optim
import torch.nn as nn

from models.unet import UNet

# from models.model1 import Model1
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

    # device = "cpu"

    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    batch_size = 16
    lr = 0.001
    epochs = 10

    train_loader, test_loader, eval_loader = data.load_data(
        train=0.80, test=0.10, eval=0.10, batch_size=batch_size
    )

    models = [
        UNet(input_channels=3, output_channels=21, device=device),
        # Model2(device=device),
        # Model3(device=device),
    ]

    for model in models:
        print(f"Training {model.model_name}...")
        # print(model)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        model.to(device)

        model.train_model(
            train_loader=train_loader,
            test_loader=test_loader,
            eval_loader=eval_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=epochs,
        )

        # model.save()

    # model.plot_train_history()
