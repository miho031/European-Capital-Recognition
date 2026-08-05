from pathlib import Path

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

IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_WORKERS = 4

LEARNING_RATE = 0.001

EPOCHS = 15

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)

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

images, labels = next(iter(train_loader))

images = images.to(DEVICE)

outputs = model(images)

print(outputs.shape)
