import numpy as np
from sklearn.model_selection import train_test_split
import torch


# STILL FAKE DATA TODO
def load_data(train=0.80, test=0.10, eval=0.10):
    # Ensure the ratios sum to 1
    if not np.isclose(train + test + eval, 1.0):
        raise ValueError("The sum of train, test, and eval ratios must be 1.0")

    X = np.random.rand(1000, 10)  # 1000 samples, 10 features
    y = np.random.randint(0, 2, size=(1000,))  # Binary classification labels

    # Split into train and temp (test + eval)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train), random_state=42
    )

    # Calculate the proportion of test and eval in the temp set
    test_ratio = test / (test + eval)

    # Split temp into test and eval
    X_test, X_eval, y_test, y_eval = train_test_split(
        X_temp, y_temp, test_size=(1 - test_ratio), random_state=42
    )

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    X_eval = torch.tensor(X_eval, dtype=torch.float32)
    y_eval = torch.tensor(y_eval, dtype=torch.float32).unsqueeze(1)

    return X_train, y_train, X_test, y_test, X_eval, y_eval
