#!/usr/bin/env python3

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


lr = 0.1
momentum = 0.9
weight_decay = 0.0005
dropout = 0.1
batch_size = 128
epochs = 20
base_channels = 64
label_smoothing = 0.1
num_workers = 4


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.residual(inputs) + self.shortcut(inputs))


class CifarResNet(nn.Module):
    def __init__(self, channels: int = base_channels) -> None:
        super().__init__()
        self.current_channels = channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self.make_layer(channels, blocks=2, stride=1)
        self.layer2 = self.make_layer(channels * 2, blocks=2, stride=2)
        self.layer3 = self.make_layer(channels * 4, blocks=2, stride=2)
        self.layer4 = self.make_layer(channels * 8, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels * 8 * BasicBlock.expansion, 10)

    def make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.current_channels, out_channels, stride)]
        self.current_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.current_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        features = self.pool(features)
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        return self.classifier(features)


def make_loaders(data_dir: Path, device: torch.device) -> tuple[DataLoader, DataLoader]:
    print(f"Preparing CIFAR-10 data in {data_dir}", flush=True)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    train_data = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    validation_data = datasets.CIFAR10(data_dir, train=False, download=True, transform=validation_transform)
    print("CIFAR-10 data ready", flush=True)
    pin_memory = device.type == "cuda"
    return (
        DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        ),
        DataLoader(
            validation_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        ),
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    use_amp = device.type == "cuda"
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = loss_fn(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_examples = labels.numel()
        total_loss += float(loss.item()) * batch_examples
        total_examples += batch_examples
    return total_loss / total_examples


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += labels.numel()
    return correct / total


def main() -> None:
    torch.manual_seed(7)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = make_loaders(Path("/tmp/autotune-cifar10-data"), device)
    model = CifarResNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(epochs):
        training_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
        validation_accuracy = evaluate(model, validation_loader, device)
        scheduler.step()
        print(
            f"epoch={epoch + 1}/{epochs} loss={training_loss:.4f} "
            f"validation_accuracy={validation_accuracy:.4f}",
            flush=True,
        )

    validation_accuracy = evaluate(model, validation_loader, device)
    _ = validation_accuracy


if __name__ == "__main__":
    main()
