import csv
import shutil
from pathlib import Path


REVIEW_ROOT = Path("dataset/review")
PROCESSED_ROOT = Path("dataset/processed")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def read_metadata(metadata_path: Path):
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

        return reader.fieldnames, list(reader)


def write_metadata(
    metadata_path: Path,
    fieldnames: list[str],
    rows: list[dict],
):
    with metadata_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def approve_city(review_city: Path) -> int:
    city_name = review_city.name

    review_images = review_city / "images"
    review_metadata = review_city / "metadata.csv"

    processed_city = PROCESSED_ROOT / city_name
    processed_images = processed_city / "images"
    processed_metadata = processed_city / "metadata.csv"

    if not review_images.exists():
        return 0

    if not review_metadata.exists():
        print(
            f"{city_name}: nema review metadata.csv."
        )
        return 0

    processed_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_fieldnames, review_rows = read_metadata(
        review_metadata
    )

    if processed_metadata.exists():

        processed_fieldnames, processed_rows = (
            read_metadata(processed_metadata)
        )

        if processed_fieldnames != review_fieldnames:
            raise ValueError(
                f"Metadata stupci se ne podudaraju za {city_name}."
            )

    else:
        processed_rows = []

    # Slike koje su ostale u review/images smatramo odobrenima.
    approved_images = sorted(
        image
        for image in review_images.iterdir()
        if (
            image.is_file()
            and image.suffix.lower() in IMAGE_EXTENSIONS
        )
    )

    approved_filenames = {
        image.name
        for image in approved_images
    }

    if not approved_filenames:
        print(
            f"{city_name}: nema slika za odobravanje."
        )
        return 0

    existing_filenames = {
        row.get("filename", "").strip()
        for row in processed_rows
    }

    rows_to_add = [
        row
        for row in review_rows
        if (
            row.get("filename", "").strip()
            in approved_filenames
            and row.get("filename", "").strip()
            not in existing_filenames
        )
    ]

    # Kopiranje slika
    for image in approved_images:

        destination = (
            processed_images
            / image.name
        )

        if not destination.exists():
            shutil.copy2(
                image,
                destination,
            )

    # Dodavanje metadata redaka
    processed_rows.extend(
        rows_to_add
    )

    write_metadata(
        processed_metadata,
        review_fieldnames,
        processed_rows,
    )

    # Nakon uspješnog odobravanja brišemo slike iz review foldera.
    for image in approved_images:
        image.unlink()

    print(
        f"{city_name}: odobreno "
        f"{len(approved_images)} slika."
    )

    return len(approved_images)


def main():
    if not REVIEW_ROOT.exists():
        raise FileNotFoundError(
            f"Review folder ne postoji: "
            f"{REVIEW_ROOT.resolve()}"
        )

    city_folders = sorted(
        folder
        for folder in REVIEW_ROOT.iterdir()
        if folder.is_dir()
    )

    total_approved = 0

    print("=" * 60)
    print("ODOBRAVANJE REVIEW SLIKA")
    print("=" * 60)

    for city_folder in city_folders:

        total_approved += approve_city(
            city_folder
        )

    print("\n" + "=" * 60)

    print(
        f"Ukupno odobreno: "
        f"{total_approved}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
