from pathlib import Path
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models


BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "data" / "test"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_model_full_large.pth"
)

OUTPUT_DIR = BASE_DIR / "error_analysis"

BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# Dataset
# =========================

weights = models.EfficientNet_B0_Weights.DEFAULT
transform = weights.transforms()

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


print(f"Device: {DEVICE}")
print(f"Broj test slika: {len(test_dataset)}")
print(f"Klase: {test_dataset.classes}")


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
# Analiza pogrešaka
# =========================

error_count = 0
sample_index = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        for i in range(len(labels)):

            true_label = labels[i].item()
            predicted_label = predictions[i].item()

            current_sample = sample_index
            sample_index += 1

            if true_label == predicted_label:
                continue

            original_path, _ = test_dataset.samples[
                current_sample
            ]

            true_class = test_dataset.classes[
                true_label
            ]

            predicted_class = test_dataset.classes[
                predicted_label
            ]

            output_folder = (
                OUTPUT_DIR
                / true_class
                / f"predicted_as_{predicted_class}"
            )

            output_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            destination = (
                output_folder
                / Path(original_path).name
            )

            shutil.copy2(
                original_path,
                destination,
            )

            error_count += 1

            print(
                f"Pogreška: "
                f"{true_class} -> {predicted_class} | "
                f"{Path(original_path).name}"
            )


# =========================
# Rezultat
# =========================

print("\n=========================")
print("ANALIZA ZAVRŠENA")
print("=========================")

print(
    f"Ukupno pogrešno klasificiranih slika: "
    f"{error_count}"
)

print(
    f"Slike spremljene u: {OUTPUT_DIR}"
)
