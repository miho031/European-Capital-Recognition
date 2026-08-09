import random
import shutil
from dataclasses import dataclass
from pathlib import Path


# ============================================================
# POSTAVKE
# ============================================================


DATASET_ROOT = Path("dataset/raw_4cities")
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
        """Vraća ukupan broj slika u klasi."""

        return len(self.train) + len(self.val) + len(self.test)


# ============================================================
# POMOĆNE FUNKCIJE
# ============================================================


def validate_ratios() -> None:
    """Provjerava da zbroj omjera za train, val i test iznosi 1."""

    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO

    if abs(total_ratio - 1.0) > 0.0001:
        raise ValueError(
            "Zbroj TRAIN_RATIO, VAL_RATIO i TEST_RATIO mora biti 1.0."
        )


def get_class_folders(dataset_root: Path) -> list[Path]:
    """Vraća sortirane mape klasa iz raw dataset direktorija."""

    if not dataset_root.exists():
        raise FileNotFoundError(f"Raw dataset mapa ne postoji: {dataset_root}")

    return sorted(
        folder
        for folder in dataset_root.iterdir()
        if folder.is_dir()
    )


def get_class_images(class_folder: Path) -> list[Path]:
    """Vraća sve podržane slike za jednu klasu."""

    images_folder = class_folder / "images"

    if not images_folder.exists():
        return []

    return sorted(
        image
        for image in images_folder.iterdir()
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_images(
    images: list[Path],
    random_generator: random.Random,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Miješa slike i dijeli ih na train, val i test dio."""

    shuffled_images = images.copy()
    random_generator.shuffle(shuffled_images)

    train_count = int(len(shuffled_images) * TRAIN_RATIO)
    val_count = int(len(shuffled_images) * VAL_RATIO)

    train_end = train_count
    val_end = train_end + val_count

    train_images = shuffled_images[:train_end]
    val_images = shuffled_images[train_end:val_end]
    test_images = shuffled_images[val_end:]

    return train_images, val_images, test_images


def build_class_splits(dataset_root: Path) -> list[ClassSplit]:
    """Gradi split za svaku klasu u raw datasetu."""

    random_generator = random.Random(RANDOM_SEED)
    class_splits: list[ClassSplit] = []

    for class_folder in get_class_folders(dataset_root):
        images = get_class_images(class_folder)

        if not images:
            print(f"Upozorenje: klasa '{class_folder.name}' nema slika.")
            continue

        train_images, val_images, test_images = split_images(
            images=images,
            random_generator=random_generator,
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


def prepare_output_folder(output_root: Path) -> None:
    """Priprema izlaznu data mapu i briše stare split direktorije."""

    output_root.mkdir(parents=True, exist_ok=True)

    for split_name in (TRAIN_DIR_NAME, VAL_DIR_NAME, TEST_DIR_NAME):
        split_folder = output_root / split_name

        if split_folder.exists():
            shutil.rmtree(split_folder)

        split_folder.mkdir(parents=True, exist_ok=True)


def copy_images(
    images: list[Path],
    destination_folder: Path,
) -> None:
    """Kopira slike u zadanu odredišnu mapu."""

    destination_folder.mkdir(parents=True, exist_ok=True)

    for image in images:
        shutil.copy2(
            image,
            destination_folder / image.name,
        )


def write_split_to_disk(
    class_splits: list[ClassSplit],
    output_root: Path,
) -> None:
    """Kopira sve splitane slike u data mapu."""

    for class_split in class_splits:
        copy_images(
            images=class_split.train,
            destination_folder=output_root / TRAIN_DIR_NAME / class_split.class_name,
        )
        copy_images(
            images=class_split.val,
            destination_folder=output_root / VAL_DIR_NAME / class_split.class_name,
        )
        copy_images(
            images=class_split.test,
            destination_folder=output_root / TEST_DIR_NAME / class_split.class_name,
        )


def count_split_images(class_splits: list[ClassSplit]) -> tuple[int, int, int]:
    """Vraća ukupan broj train, val i test slika."""

    train_total = sum(len(class_split.train) for class_split in class_splits)
    val_total = sum(len(class_split.val) for class_split in class_splits)
    test_total = sum(len(class_split.test) for class_split in class_splits)

    return train_total, val_total, test_total


def print_statistics(class_splits: list[ClassSplit]) -> None:
    """Ispisuje jasnu statistiku splitanja dataseta."""

    train_total, val_total, test_total = count_split_images(class_splits)
    total_images = train_total + val_total + test_total

    print("\n===== Dataset split statistika =====")
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
    print(f"{'Klasa':<15} {'Train':>7} {'Val':>7} {'Test':>7} {'Ukupno':>8}")
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
    print("\nSplitanje je završeno.")


# ============================================================
# GLAVNI PROGRAM
# ============================================================


def main() -> None:
    """Pokreće izradu train, val i test dataseta."""

    validate_ratios()
    class_splits = build_class_splits(DATASET_ROOT)

    if not class_splits:
        raise RuntimeError("Nije pronađena nijedna klasa sa slikama.")

    prepare_output_folder(OUTPUT_ROOT)
    write_split_to_disk(
        class_splits=class_splits,
        output_root=OUTPUT_ROOT,
    )
    print_statistics(class_splits)


if __name__ == "__main__":
    main()
