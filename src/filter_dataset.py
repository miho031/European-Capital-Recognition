import csv
import hashlib
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


# ============================================================
# POSTAVKE
# ============================================================

INPUT_ROOT = Path("dataset/raw")
OUTPUT_ROOT = Path("dataset/processed")
REVIEW_ROOT = Path("dataset/review")

REPORT_PATH = Path("filter_report.csv")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Minimalna dopuštena dimenzija slike.
# Ovo je namjerno dosta blago.
MIN_WIDTH = 300
MIN_HEIGHT = 200

# Ekstremno tamne / svijetle slike.
# Prosječna vrijednost piksela je 0-255.
MIN_MEAN_BRIGHTNESS = 8
MAX_MEAN_BRIGHTNESS = 247

# Blur threshold koristimo samo za označavanje slika.
# Slike se zbog blura NE brišu automatski.
BLUR_REVIEW_THRESHOLD = 35.0

# Hamming distance za perceptual hash.
# Koristimo samo kao signal za pregled.
NEAR_DUPLICATE_DISTANCE = 3


# ============================================================
# HASH FUNKCIJE
# ============================================================


def calculate_file_hash(image_path: Path) -> str:
    """
    SHA-256 hash za pronalaženje potpuno identičnih datoteka.
    """

    sha256 = hashlib.sha256()

    with image_path.open("rb") as file:
        while True:
            chunk = file.read(8192)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def calculate_dhash(image: Image.Image) -> int:
    """
    Jednostavan perceptual difference hash (dHash).

    Koristi se za pronalaženje vizualno vrlo sličnih slika.
    """

    gray = image.convert("L").resize(
        (9, 8),
        Image.Resampling.LANCZOS,
    )

    pixels = np.asarray(gray, dtype=np.int16)

    difference = pixels[:, 1:] > pixels[:, :-1]

    hash_value = 0

    for bit in difference.flatten():
        hash_value = (hash_value << 1) | int(bit)

    return hash_value


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """
    Broj različitih bitova između dva perceptual hasha.
    """

    return (hash_a ^ hash_b).bit_count()


# ============================================================
# ANALIZA KVALITETE
# ============================================================


def calculate_blur_score(image: Image.Image) -> float:
    """
    Procjenjuje oštrinu slike pomoću varijance
    diskretne Laplaceove aproksimacije.

    Manja vrijednost = mutnija slika.

    Rezultat koristimo samo kao indikator za ručni pregled.
    """

    gray = image.convert("L")

    # Smanjujemo sliku radi bržeg računanja.
    gray.thumbnail((800, 800))

    array = np.asarray(gray, dtype=np.float32)

    if array.shape[0] < 3 or array.shape[1] < 3:
        return 0.0

    center = array[1:-1, 1:-1]

    laplacian = (
        -4 * center
        + array[:-2, 1:-1]
        + array[2:, 1:-1]
        + array[1:-1, :-2]
        + array[1:-1, 2:]
    )

    return float(np.var(laplacian))


def calculate_brightness(image: Image.Image) -> float:
    """
    Vraća prosječnu svjetlinu slike u rasponu 0-255.
    """

    gray = image.convert("L")
    gray.thumbnail((500, 500))

    array = np.asarray(gray, dtype=np.float32)

    return float(array.mean())


# ============================================================
# METADATA
# ============================================================


def load_metadata(metadata_path: Path) -> tuple[list[str], list[dict]]:
    """
    Učitava metadata.csv.
    """

    with metadata_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError(
                f"Metadata nema zaglavlje: {metadata_path}"
            )

        rows = list(reader)

        return reader.fieldnames, rows


def write_filtered_metadata(
    destination_path: Path,
    fieldnames: list[str],
    rows: list[dict],
    accepted_filenames: set[str],
) -> None:
    """
    Kopira u processed metadata samo zapise slika
    koje su prošle filtriranje.
    """

    filtered_rows = [
        row
        for row in rows
        if row.get("filename", "").strip()
        in accepted_filenames
    ]

    with destination_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(filtered_rows)


# ============================================================
# FILTRIRANJE JEDNOG GRADA
# ============================================================


def process_city(city_folder: Path) -> list[dict]:
    city_name = city_folder.name

    images_folder = city_folder / "images"
    metadata_path = city_folder / "metadata.csv"

    destination_city = OUTPUT_ROOT / city_name
    destination_images = destination_city / "images"

    destination_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print(f"GRAD: {city_name}")
    print("=" * 60)

    if not images_folder.exists():
        print("Nema images direktorija.")
        return []

    if not metadata_path.exists():
        print("Nema metadata.csv.")
        return []

    fieldnames, metadata_rows = load_metadata(
        metadata_path
    )

    if "filename" not in fieldnames:
        raise ValueError(
            f"{metadata_path} nema stupac 'filename'."
        )

    image_paths = sorted(
        image_path
        for image_path in images_folder.iterdir()
        if (
            image_path.is_file()
            and image_path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    report_rows = []

    accepted_filenames = set()

    review_filenames = set()

    review_city = REVIEW_ROOT / city_name
    review_images = review_city / "images"

    review_images.mkdir(
        parents=True,
        exist_ok=True,
    )
    # Exact duplicate hashovi
    exact_hashes = {}

    # Perceptual hashovi prihvaćenih slika
    perceptual_hashes = []

    counters = Counter()

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        filename = image_path.name

        status = "accepted"
        reasons = []

        width = None
        height = None
        brightness = None
        blur_score = None

        try:
            # Prvo provjeravamo da Pillow može otvoriti datoteku.
            with Image.open(image_path) as image:
                image.verify()

            # Ponovno otvaramo nakon verify().
            with Image.open(image_path) as image:

                image = image.convert("RGB")

                width, height = image.size

                # --------------------------------------------
                # PREMALA SLIKA
                # --------------------------------------------

                if (
                    width < MIN_WIDTH
                    or height < MIN_HEIGHT
                ):
                    status = "rejected"
                    reasons.append("too_small")

                # --------------------------------------------
                # SVJETLINA
                # --------------------------------------------

                brightness = calculate_brightness(
                    image
                )

                if (
                    brightness
                    < MIN_MEAN_BRIGHTNESS
                ):
                    status = "rejected"
                    reasons.append("extremely_dark")

                elif (
                    brightness
                    > MAX_MEAN_BRIGHTNESS
                ):
                    status = "rejected"
                    reasons.append("extremely_bright")

                # --------------------------------------------
                # BLUR
                # --------------------------------------------

                blur_score = calculate_blur_score(
                    image
                )

                if (
                    blur_score
                    < BLUR_REVIEW_THRESHOLD
                    and status != "rejected"
                ):
                    status = "review"
                    reasons.append("possible_blur")

                # --------------------------------------------
                # EXACT DUPLICATE
                # --------------------------------------------

                file_hash = calculate_file_hash(
                    image_path
                )

                if file_hash in exact_hashes:

                    status = "rejected"

                    duplicate_of = (
                        exact_hashes[file_hash]
                    )

                    reasons.append(
                        f"exact_duplicate_of:"
                        f"{duplicate_of}"
                    )

                else:
                    exact_hashes[file_hash] = (
                        filename
                    )

                # --------------------------------------------
                # NEAR DUPLICATE
                # --------------------------------------------

                perceptual_hash = calculate_dhash(
                    image
                )

                near_duplicate_of = None

                for (
                    previous_filename,
                    previous_hash,
                ) in perceptual_hashes:

                    distance = hamming_distance(
                        perceptual_hash,
                        previous_hash,
                    )

                    if (
                        distance
                        <= NEAR_DUPLICATE_DISTANCE
                    ):
                        near_duplicate_of = (
                            previous_filename
                        )
                        break

                if (
                    near_duplicate_of
                    and status != "rejected"
                ):
                    # Ne brišemo automatski.
                    # Samo označavamo za pregled.
                    status = "review"

                    reasons.append(
                        "possible_near_duplicate_of:"
                        f"{near_duplicate_of}"
                    )

                if status != "rejected":
                    perceptual_hashes.append(
                        (
                            filename,
                            perceptual_hash,
                        )
                    )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as error:

            status = "rejected"

            reasons.append(
                f"corrupted_or_invalid:"
                f"{type(error).__name__}"
            )

        # ----------------------------------------------------
        # KOPIRANJE
        # ----------------------------------------------------

        if status == "accepted":

            shutil.copy2(
                image_path,
                destination_images / filename,
            )

            accepted_filenames.add(filename)

        elif status == "review":

            shutil.copy2(
                image_path,
                review_images / filename,
            )

            review_filenames.add(filename)

        counters[status] += 1

        report_rows.append(
            {
                "city": city_name,
                "filename": filename,
                "status": status,
                "reason": ";".join(reasons),
                "width": width,
                "height": height,
                "brightness": (
                    f"{brightness:.2f}"
                    if brightness is not None
                    else ""
                ),
                "blur_score": (
                    f"{blur_score:.2f}"
                    if blur_score is not None
                    else ""
                ),
            }
        )

        if index % 50 == 0:
            print(
                f"Obrađeno "
                f"{index}/{len(image_paths)}..."
            )

    # --------------------------------------------------------
    # NOVI METADATA.CSV
    # --------------------------------------------------------

    write_filtered_metadata(
        destination_city / "metadata.csv",
        fieldnames,
        metadata_rows,
        accepted_filenames,
    )

    write_filtered_metadata(
        review_city / "metadata.csv",
        fieldnames,
        metadata_rows,
        review_filenames,
    )

    print()
    print(f"Ukupno:    {len(image_paths)}")
    print(f"Prihvaćeno: {counters['accepted']}")
    print(f"Za pregled: {counters['review']}")
    print(f"Odbačeno:   {counters['rejected']}")

    return report_rows


# ============================================================
# REPORT
# ============================================================


def write_report(report_rows: list[dict]) -> None:

    fieldnames = [
        "city",
        "filename",
        "status",
        "reason",
        "width",
        "height",
        "brightness",
        "blur_score",
    ]

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(report_rows)


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Input dataset ne postoji: "
            f"{INPUT_ROOT.resolve()}"
        )

    if REVIEW_ROOT.exists():
        shutil.rmtree(REVIEW_ROOT)

    REVIEW_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    # Brišemo stari processed dataset.
    # RAW DATASET SE NE DIRA.
    if OUTPUT_ROOT.exists():
        print(
            f"Brisanje starog processed dataseta: "
            f"{OUTPUT_ROOT}"
        )

        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    city_folders = sorted(
        folder
        for folder in INPUT_ROOT.iterdir()
        if folder.is_dir()
    )

    all_report_rows = []

    print("=" * 60)
    print("DATASET QUALITY FILTER")
    print("=" * 60)

    for city_folder in city_folders:

        city_report = process_city(
            city_folder
        )

        all_report_rows.extend(
            city_report
        )

    write_report(
        all_report_rows
    )

    # --------------------------------------------------------
    # UKUPNA STATISTIKA
    # --------------------------------------------------------

    status_counts = Counter(
        row["status"]
        for row in all_report_rows
    )

    print("\n" + "=" * 60)
    print("UKUPNA STATISTIKA")
    print("=" * 60)

    print(
        f"Ukupno slika: "
        f"{len(all_report_rows)}"
    )

    print(
        f"Prihvaćeno: "
        f"{status_counts['accepted']}"
    )

    print(
        f"Za pregled: "
        f"{status_counts['review']}"
    )

    print(
        f"Odbačeno: "
        f"{status_counts['rejected']}"
    )

    print(
        f"\nIzvještaj spremljen u: "
        f"{REPORT_PATH}"
    )

    print(
        f"Filtrirani dataset: "
        f"{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()
