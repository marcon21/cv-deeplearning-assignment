import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import Subset, DataLoader
import os


# Define a top-level identity transform to avoid lambda pickling issues
class IdentityTransform:
    def __call__(self, x):
        return x


def collate_fn(batch):
    images = []
    masks = []
    for img, mask in batch:
        images.append(img)
        # Remove channel dimension if present (should be [1, H, W])
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        masks.append(mask)
    return torch.stack(images), torch.stack(masks)


def load_data(
    train=0.80,
    test=0.10,
    eval=0.10,
    root_dir="./VOC",
    batch_size=32,
    num_workers=4,
    grayscale=False,
):
    # Ensure the ratios sum to 1
    if not np.isclose(train + test + eval, 1.0):
        raise ValueError("The sum of train, test, and eval ratios must be 1.0")

    # Enhanced transforms with normalization
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet stats
            (
                transforms.Grayscale(num_output_channels=1)
                if grayscale
                else IdentityTransform()
            ),
        ]
    )

    target_transform = transforms.Compose(
        [
            transforms.Resize(
                (256, 256), interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.PILToTensor(),
        ]
    )

    # Automatically set download flag based on presence of extracted data
    voc_root = os.path.join(root_dir, "VOCdevkit", "VOC2012")
    download = not os.path.exists(voc_root)

    dataset = VOCSegmentation(
        root=root_dir,
        year="2012",
        image_set="trainval",
        download=download,
        transform=transform,
        target_transform=target_transform,
    )

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_end = int(train * dataset_size)
    test_end = train_end + int(test * dataset_size)

    train_indices = indices[:train_end]
    test_indices = indices[train_end:test_end]
    eval_indices = indices[test_end:]

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    eval_subset = Subset(dataset, eval_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,  # Use custom collate function
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,  # Use custom collate function
    )
    eval_loader = DataLoader(
        eval_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,  # Use custom collate function
    )

    return train_loader, test_loader, eval_loader


if __name__ == "__main__":
    # Example usage: iterate over the train dataloader
    train_loader, test_loader, eval_loader = load_data()
    for images, masks in train_loader:
        print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")
        break
