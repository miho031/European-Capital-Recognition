from pathlib import Path
from py_compile import main
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ===========================
# Konstante
# ===========================

DATA_ROOT = Path("data")

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(
    exist_ok=True
)

IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_WORKERS = 4

LEARNING_RATE = 0.001

EPOCHS = 15

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

weights = models.EfficientNet_B0_Weights.DEFAULT

train_transform = weights.transforms()

val_transform = weights.transforms()

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []


def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(
        train_loader,
        desc="Training",
    ):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(accuracy)
    return epoch_loss, accuracy


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(
            val_loader,
            desc="Validation",
        ):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    val_losses.append(val_loss)
    val_accuracies.append(accuracy)

    return val_loss, accuracy


def main():
    print(f"Device: {DEVICE}")
    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    num_classes = len(train_dataset.classes)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )

    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_accuracy = 0

    for epoch in range(EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Epoch {epoch+1}/{EPOCHS}"
        )
        print(
            f"Train loss: {train_loss:.4f}"
        )
        print(
            f"Train accuracy: {train_acc:.2f}%"
        )
        print(
            f"Validation loss: {val_loss:.4f}"
        )
        print(
            f"Validation accuracy: {val_acc:.2f}%"
        )

    print(f"Broj klasa: {len(train_dataset.classes)}")
    print(train_dataset.classes)

    print(f"Broj train slika: {len(train_dataset)}")
    print(f"Broj validation slika: {len(val_dataset)}")

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        torch.save(
            model.state_dict(),
            MODELS_DIR / "best_model.pth",
        )

        print("Model spremljen.")


if __name__ == "__main__":
    main()
