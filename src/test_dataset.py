from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
except ModuleNotFoundError as error:
    raise SystemExit(
        "Nedostaje PyTorch dependency. Instaliraj ga naredbom: "
        "py -m pip install -r requirements.txt"
    ) from error


# ============================================================
# POSTAVKE
# ============================================================


DATA_ROOT = Path("data")

TRAIN_DIR_NAME = "train"
VAL_DIR_NAME = "val"
TEST_DIR_NAME = "test"

EXPECTED_NUM_CLASSES = 15
EXPECTED_SPLIT_COUNTS = {
    TRAIN_DIR_NAME: 857,
    VAL_DIR_NAME: 179,
    TEST_DIR_NAME: 199,
}

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0


# ============================================================
# POMOĆNE FUNKCIJE
# ============================================================


def build_transform() -> transforms.Compose:
    """Vraća transformacije koje pripremaju slike za PyTorch model."""

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_dataset(
    split_name: str,
    transform: transforms.Compose,
) -> datasets.ImageFolder:
    """Učitava jedan dataset split pomoću ImageFolder strukture."""

    split_path = DATA_ROOT / split_name

    if not split_path.exists():
        raise FileNotFoundError(f"Split mapa ne postoji: {split_path}")

    return datasets.ImageFolder(
        root=split_path,
        transform=transform,
    )


def validate_num_classes(dataset: datasets.ImageFolder) -> None:
    """Provjerava da dataset sadrži očekivani broj klasa."""

    actual_num_classes = len(dataset.classes)

    if actual_num_classes != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"Očekivano je {EXPECTED_NUM_CLASSES} klasa, "
            f"ali pronađeno je {actual_num_classes}."
        )


def validate_split_count(
    split_name: str,
    dataset: datasets.ImageFolder,
) -> None:
    """Provjerava da split ima očekivani broj slika."""

    expected_count = EXPECTED_SPLIT_COUNTS[split_name]
    actual_count = len(dataset)

    if actual_count != expected_count:
        raise ValueError(
            f"Split '{split_name}' treba imati {expected_count} slika, "
            f"ali ima {actual_count}."
        )


def validate_class_names(reference_classes: list[str], classes: list[str]) -> None:
    """Provjerava da svi splitovi imaju isti popis klasa."""

    if classes != reference_classes:
        raise ValueError("Splitovi nemaju isti popis klasa.")


def validate_batch(
    split_name: str,
    dataset: datasets.ImageFolder,
) -> tuple[torch.Size, torch.Size]:
    """Učitava jedan batch i provjerava da transformacije rade."""

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    images, labels = next(iter(data_loader))

    expected_shape = (
        min(BATCH_SIZE, len(dataset)),
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    if tuple(images.shape) != expected_shape:
        raise ValueError(
            f"Split '{split_name}' ima neočekivan oblik batcha: "
            f"{tuple(images.shape)}."
        )

    if labels.ndim != 1:
        raise ValueError(f"Split '{split_name}' ima neočekivan oblik labela.")

    return images.shape, labels.shape


def print_split_statistics(
    split_name: str,
    dataset: datasets.ImageFolder,
    image_shape: torch.Size,
    label_shape: torch.Size,
) -> None:
    """Ispisuje statistiku za jedan dataset split."""

    print(
        f"{split_name:<6} "
        f"slike={len(dataset):>4} "
        f"klase={len(dataset.classes):>2} "
        f"batch_slike={tuple(image_shape)} "
        f"batch_labele={tuple(label_shape)}"
    )


# ============================================================
# GLAVNI PROGRAM
# ============================================================


def main() -> None:
    """Pokreće osnovnu provjeru PyTorch dataseta."""

    transform = build_transform()
    split_names = [TRAIN_DIR_NAME, VAL_DIR_NAME, TEST_DIR_NAME]

    datasets_by_split = {
        split_name: load_dataset(
            split_name=split_name,
            transform=transform,
        )
        for split_name in split_names
    }

    reference_classes = datasets_by_split[TRAIN_DIR_NAME].classes

    print("===== PyTorch dataset provjera =====")
    print(f"Data mapa: {DATA_ROOT}")
    print(f"Očekivani broj klasa: {EXPECTED_NUM_CLASSES}")
    print(f"Klase: {', '.join(reference_classes)}")
    print()

    for split_name, dataset in datasets_by_split.items():
        validate_num_classes(dataset)
        validate_split_count(
            split_name=split_name,
            dataset=dataset,
        )
        validate_class_names(
            reference_classes=reference_classes,
            classes=dataset.classes,
        )
        image_shape, label_shape = validate_batch(
            split_name=split_name,
            dataset=dataset,
        )
        print_split_statistics(
            split_name=split_name,
            dataset=dataset,
            image_shape=image_shape,
            label_shape=label_shape,
        )

    print("\nSve provjere su prošle.")


if __name__ == "__main__":
    main()
