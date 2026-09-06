from pathlib import Path
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

train_transform = transforms.Compose([
    # Nasumično izreže dio slike i skalira ga na 224x224.
    # scale=(0.80, 1.0) znači da neće raditi ekstremne cropove.
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.80, 1.0),
    ),

    # Male promjene osvjetljenja i boje.
    # Korisno jer Mapillary slike nastaju u različitim
    # vremenskim i svjetlosnim uvjetima.
    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.15,
        hue=0.02,
    ),

    # Vrlo mala rotacija radi otpornosti na nagib kamere.
    transforms.RandomRotation(
        degrees=5,
    ),

    transforms.ToTensor(),

    # ImageNet normalizacija koju očekuje pretrained EfficientNet.
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Validation skup se NE augmentira.
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

    # Odmrzni sve slojeve EfficientNeta
    for param in model.features.parameters():
        param.requires_grad = True

    num_classes = len(train_dataset.classes)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )

    # Otključaj zadnji blok EfficientNeta
    # for param in model.features[-1].parameters():
    #     param.requires_grad = True

    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Ukupno parametara: {total_params:,}")
    print(f"Parametara za treniranje: {trainable_params:,}")
    print(
        f"Postotak parametara koji se trenira: "
        f"{100 * trainable_params / total_params:.2f}%"
    )

    best_accuracy = 0.0
    best_epoch = 0

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
            f"Epoch {epoch + 1}/{EPOCHS}"
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

        # Sprema model čim postigne najbolji validation rezultat.
        if val_acc > best_accuracy:

            best_accuracy = val_acc
            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                MODELS_DIR / "best_model_full_augmented.pth",
            )

            print(
                f"Novi najbolji model spremljen "
                f"({best_accuracy:.2f}%)."
            )

    print(f"\nNajbolji epoch: {best_epoch}")
    print(
        f"Najbolja validation accuracy: "
        f"{best_accuracy:.2f}%"
    )

    print(f"Broj klasa: {len(train_dataset.classes)}")
    print(train_dataset.classes)

    print(f"Broj train slika: {len(train_dataset)}")
    print(
        f"Broj validation slika: "
        f"{len(val_dataset)}"
    )


if __name__ == "__main__":
    main()
