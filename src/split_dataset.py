import csv
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


# ============================================================
# POSTAVKE
# ============================================================

DATASET_ROOT = Path("dataset/processed")
OUTPUT_ROOT = Path("data")

TRAIN_DIR_NAME = "train"
VAL_DIR_NAME = "val"
TEST_DIR_NAME = "test"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ============================================================
# MODELI PODATAKA
# ============================================================


@dataclass(frozen=True)
class ClassSplit:
    """Predstavlja podjelu slika jedne klase na train, val i test skup."""

    class_name: str
    train: list[Path]
    val: list[Path]
    test: list[Path]

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)


# ============================================================
# POMOĆNE FUNKCIJE
# ============================================================


def validate_ratios() -> None:
    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO

    if abs(total_ratio - 1.0) > 0.0001:
        raise ValueError(
            "Zbroj TRAIN_RATIO, VAL_RATIO i TEST_RATIO mora biti 1.0."
        )


def get_class_folders(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Raw dataset mapa ne postoji: {dataset_root}"
        )

    return sorted(
        folder
        for folder in dataset_root.iterdir()
        if folder.is_dir()
    )


def load_sequences(class_folder: Path) -> dict[str, list[Path]]:
    """
    Čita metadata.csv i grupira slike prema sequence_id.
    """

    metadata_path = class_folder / "metadata.csv"
    images_folder = class_folder / "images"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.csv ne postoji za klasu: {class_folder.name}"
        )

    if not images_folder.exists():
        raise FileNotFoundError(
            f"images mapa ne postoji za klasu: {class_folder.name}"
        )

    sequences: dict[str, list[Path]] = defaultdict(list)

    with metadata_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        if "filename" not in reader.fieldnames:
            raise ValueError(
                f"metadata.csv za {class_folder.name} "
                "nema stupac 'filename'."
            )

        if "sequence_id" not in reader.fieldnames:
            raise ValueError(
                f"metadata.csv za {class_folder.name} "
                "nema stupac 'sequence_id'."
            )

        for row in reader:
            filename = row["filename"].strip()
            sequence_id = row["sequence_id"].strip()

            if not filename:
                continue

            image_path = images_folder / filename

            if not image_path.exists():
                print(
                    f"Upozorenje: slika ne postoji: {image_path}"
                )
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            if not sequence_id:
                # Ako neka slika ipak nema sequence_id,
                # tretiramo je kao vlastitu sekvencu.
                sequence_id = f"missing_{filename}"

            sequences[sequence_id].append(image_path)

    return dict(sequences)


def split_sequences(
    sequences: dict[str, list[Path]],
    random_generator: random.Random,
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Dijeli cijele Mapillary sekvence u train, val i test.

    Cilj je približiti se omjeru 70/15/15 prema broju slika,
    bez dijeljenja iste sekvence između skupova.
    """

    sequence_items = list(sequences.items())

    # Randomizacija služi kao tie-break za sekvence iste veličine.
    random_generator.shuffle(sequence_items)

    # Veće sekvence obrađujemo prve radi boljeg balansiranja.
    sequence_items.sort(
        key=lambda item: len(item[1]),
        reverse=True,
    )

    total_images = sum(
        len(images)
        for _, images in sequence_items
    )

    targets = {
        TRAIN_DIR_NAME: total_images * TRAIN_RATIO,
        VAL_DIR_NAME: total_images * VAL_RATIO,
        TEST_DIR_NAME: total_images * TEST_RATIO,
    }

    split_sequences_map = {
        TRAIN_DIR_NAME: [],
        VAL_DIR_NAME: [],
        TEST_DIR_NAME: [],
    }

    split_counts = {
        TRAIN_DIR_NAME: 0,
        VAL_DIR_NAME: 0,
        TEST_DIR_NAME: 0,
    }

    for sequence_id, images in sequence_items:

        deficits = {
            split_name: (
                targets[split_name]
                - split_counts[split_name]
            )
            for split_name in split_counts
        }

        best_split = max(
            deficits,
            key=deficits.get,
        )

        split_sequences_map[best_split].append(
            (sequence_id, images)
        )

        split_counts[best_split] += len(images)

    train_images = [
        image
        for _, images in split_sequences_map[TRAIN_DIR_NAME]
        for image in images
    ]

    val_images = [
        image
        for _, images in split_sequences_map[VAL_DIR_NAME]
        for image in images
    ]

    test_images = [
        image
        for _, images in split_sequences_map[TEST_DIR_NAME]
        for image in images
    ]

    return train_images, val_images, test_images


def build_class_splits(
    dataset_root: Path,
) -> list[ClassSplit]:

    random_generator = random.Random(RANDOM_SEED)
    class_splits: list[ClassSplit] = []

    for class_folder in get_class_folders(dataset_root):

        sequences = load_sequences(class_folder)

        if not sequences:
            print(
                f"Upozorenje: klasa '{class_folder.name}' "
                "nema dostupnih slika."
            )
            continue

        train_images, val_images, test_images = (
            split_sequences(
                sequences=sequences,
                random_generator=random_generator,
            )
        )

        class_splits.append(
            ClassSplit(
                class_name=class_folder.name,
                train=train_images,
                val=val_images,
                test=test_images,
            )
        )

    return class_splits


def prepare_output_folder(
    output_root: Path,
) -> None:

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split_name in (
        TRAIN_DIR_NAME,
        VAL_DIR_NAME,
        TEST_DIR_NAME,
    ):

        split_folder = output_root / split_name

        if split_folder.exists():
            shutil.rmtree(split_folder)

        split_folder.mkdir(
            parents=True,
            exist_ok=True,
        )


def copy_images(
    images: list[Path],
    destination_folder: Path,
) -> None:

    destination_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    for image in images:
        shutil.copy2(
            image,
            destination_folder / image.name,
        )


def write_split_to_disk(
    class_splits: list[ClassSplit],
    output_root: Path,
) -> None:

    for class_split in class_splits:

        copy_images(
            class_split.train,
            output_root
            / TRAIN_DIR_NAME
            / class_split.class_name,
        )

        copy_images(
            class_split.val,
            output_root
            / VAL_DIR_NAME
            / class_split.class_name,
        )

        copy_images(
            class_split.test,
            output_root
            / TEST_DIR_NAME
            / class_split.class_name,
        )


def count_split_images(
    class_splits: list[ClassSplit],
) -> tuple[int, int, int]:

    train_total = sum(
        len(class_split.train)
        for class_split in class_splits
    )

    val_total = sum(
        len(class_split.val)
        for class_split in class_splits
    )

    test_total = sum(
        len(class_split.test)
        for class_split in class_splits
    )

    return train_total, val_total, test_total


def verify_no_sequence_leakage(
    class_splits: list[ClassSplit],
) -> None:
    """
    Provjerava da ista sequence_id ne postoji u više skupova.
    """

    for class_split in class_splits:

        class_folder = (
            DATASET_ROOT
            / class_split.class_name
        )

        sequences = load_sequences(class_folder)

        image_to_sequence: dict[str, str] = {}

        for sequence_id, images in sequences.items():
            for image in images:
                image_to_sequence[image.name] = sequence_id

        seen_sequences: dict[str, str] = {}

        split_groups = {
            TRAIN_DIR_NAME: class_split.train,
            VAL_DIR_NAME: class_split.val,
            TEST_DIR_NAME: class_split.test,
        }

        for split_name, images in split_groups.items():

            for image in images:

                sequence_id = image_to_sequence.get(
                    image.name
                )

                if sequence_id is None:
                    continue

                if sequence_id in seen_sequences:

                    previous_split = (
                        seen_sequences[sequence_id]
                    )

                    if previous_split != split_name:

                        raise RuntimeError(
                            "DATA LEAKAGE: "
                            f"{class_split.class_name} | "
                            f"sekvenca {sequence_id} "
                            f"nalazi se u {previous_split} "
                            f"i {split_name}."
                        )

                else:
                    seen_sequences[sequence_id] = (
                        split_name
                    )


def print_statistics(
    class_splits: list[ClassSplit],
) -> None:

    train_total, val_total, test_total = (
        count_split_images(class_splits)
    )

    total_images = (
        train_total
        + val_total
        + test_total
    )

    print(
        "\n===== Dataset split statistika ====="
    )

    print(f"Raw mapa: {DATASET_ROOT}")
    print(f"Izlazna mapa: {OUTPUT_ROOT}")
    print(f"Seed: {RANDOM_SEED}")

    print(
        "Omjeri: "
        f"train={TRAIN_RATIO:.0%}, "
        f"val={VAL_RATIO:.0%}, "
        f"test={TEST_RATIO:.0%}"
    )

    print()

    print(
        f"{'Klasa':<15} "
        f"{'Train':>7} "
        f"{'Val':>7} "
        f"{'Test':>7} "
        f"{'Ukupno':>8}"
    )

    print("-" * 48)

    for class_split in class_splits:

        print(
            f"{class_split.class_name:<15} "
            f"{len(class_split.train):>7} "
            f"{len(class_split.val):>7} "
            f"{len(class_split.test):>7} "
            f"{class_split.total:>8}"
        )

    print("-" * 48)

    print(
        f"{'UKUPNO':<15} "
        f"{train_total:>7} "
        f"{val_total:>7} "
        f"{test_total:>7} "
        f"{total_images:>8}"
    )

    print()

    print(
        "Provjera uspješna: "
        "nijedna Mapillary sekvenca nije "
        "podijeljena između skupova."
    )

    print(
        "\nSplitanje je završeno."
    )


# ============================================================
# GLAVNI PROGRAM
# ============================================================


def main() -> None:

    validate_ratios()

    class_splits = build_class_splits(
        DATASET_ROOT
    )

    if not class_splits:
        raise RuntimeError(
            "Nije pronađena nijedna klasa sa slikama."
        )

    verify_no_sequence_leakage(
        class_splits
    )

    prepare_output_folder(
        OUTPUT_ROOT
    )

    write_split_to_disk(
        class_splits=class_splits,
        output_root=OUTPUT_ROOT,
    )

    print_statistics(
        class_splits
    )


if __name__ == "__main__":
    main()
