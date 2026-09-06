import csv
from pathlib import Path


DATASET_ROOT = Path("dataset/processed")


def sync_city(city_folder: Path):
    images_folder = city_folder / "images"
    metadata_path = city_folder / "metadata.csv"

    if not images_folder.exists() or not metadata_path.exists():
        return

    existing_filenames = {
        image.name
        for image in images_folder.iterdir()
        if image.is_file()
    }

    with metadata_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:

        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if fieldnames is None:
        return

    original_count = len(rows)

    filtered_rows = [
        row
        for row in rows
        if row.get("filename", "").strip()
        in existing_filenames
    ]

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
        writer.writerows(filtered_rows)

    removed = original_count - len(filtered_rows)

    print(
        f"{city_folder.name}: "
        f"{len(filtered_rows)} slika u metadata, "
        f"uklonjeno {removed} redaka"
    )


def main():
    city_folders = sorted(
        folder
        for folder in DATASET_ROOT.iterdir()
        if folder.is_dir()
    )

    for city_folder in city_folders:
        sync_city(city_folder)

    print("\nMetadata sinkroniziran s postojećim slikama.")


if __name__ == "__main__":
    main()
