from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# =========================
# Postavke
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "data" / "test"
MODEL_PATH = BASE_DIR / "models" / "best_model_full_large.pth"

BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {DEVICE}")


# =========================
# Transformacije
# =========================

weights = models.EfficientNet_B0_Weights.DEFAULT

transform = weights.transforms()


# =========================
# Dataset
# =========================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print(f"Broj test slika: {len(test_dataset)}")
print(f"Broj klasa: {len(test_dataset.classes)}")
print(test_dataset.classes)


# =========================
# Model
# =========================

model = models.efficientnet_b0(
    weights=None
)

num_classes = len(test_dataset.classes)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes,
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )
)

model.to(DEVICE)
model.eval()


# =========================
# Testiranje
# =========================

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# =========================
# Rezultati
# =========================

accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

print("\n=========================")
print("REZULTATI TESTIRANJA")
print("=========================")

print(
    f"Test accuracy: {accuracy * 100:.2f}%"
)


# =========================
# Classification report
# =========================

print("\nClassification report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        digits=4,
    )
)


# =========================
# Confusion matrix
# =========================

cm = confusion_matrix(
    all_labels,
    all_predictions,
)

print("\nConfusion matrix:")

print(cm)
