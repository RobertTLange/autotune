#!/usr/bin/env python3

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


lr = 0.001
dropout = 0.2
batch_size = 64
epochs = 1
train_examples = 1024
validation_examples = 512


class SmallMnistCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(16 * 7 * 7, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def make_loaders(data_dir: Path) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    validation_data = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_subset = Subset(train_data, range(train_examples))
    validation_subset = Subset(validation_data, range(validation_examples))
    return (
        DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(validation_subset, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def train_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += labels.numel()
    return correct / total


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = make_loaders(Path("/tmp/autotune-mnist-data"))
    model = SmallMnistCnn().to(device)
    for _ in range(epochs):
        train_epoch(model, train_loader, device)
    validation_accuracy = evaluate(model, validation_loader, device)
    _ = validation_accuracy


if __name__ == "__main__":
    main()
