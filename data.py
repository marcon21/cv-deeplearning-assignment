import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import Subset, DataLoader
import xml.etree.ElementTree as ET


def collate_fn(batch):
    """Process batch data to have consistent dimensions and extract annotations."""
    images = []
    targets = []

    for sample in batch:
        img, annotation = sample
        # Process the image
        images.append(img)

        # Process the annotation (XML to dictionary)
        target = {}
        anno = annotation["annotation"]
        boxes = []
        labels = []

        for obj in anno.get("object", []):
            bbox = obj["bndbox"]
            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj["name"])

        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        target["labels"] = labels
        targets.append(target)

    # return images, targets
    return torch.stack(images), targets


def load_data(
    train=0.80,
    test=0.10,
    eval=0.10,
    root_dir="./VOC",
    batch_size=32,
    num_workers=4,
):
    # Ensure the ratios sum to 1
    if not np.isclose(train + test + eval, 1.0):
        raise ValueError("The sum of train, test, and eval ratios must be 1.0")

    # Enhanced transforms with normalization
    transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),  # Consistent size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet stats
        ]
    )

    dataset = VOCDetection(
        root=root_dir,
        year="2012",
        image_set="trainval",
        download=True,
        transform=transform,
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
    for batch in train_loader:
        print(f"Batch: {batch}")
        break
