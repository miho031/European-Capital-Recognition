import csv
from collections import Counter
from pathlib import Path


DATASET_ROOT = Path("dataset/raw")


def analyze_city(city_folder: Path) -> None:
    city_name = city_folder.name
    metadata_path = city_folder / "metadata.csv"

    if not metadata_path.exists():
        print(f"\n{city_name}: metadata.csv nije pronađen.")
        return

    sequence_counts = Counter()
    total_images = 0
    missing_sequence = 0

    with metadata_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        for row in reader:
            total_images += 1

            sequence_id = row.get("sequence_id", "").strip()

            if not sequence_id:
                missing_sequence += 1
            else:
                sequence_counts[sequence_id] += 1

    print("\n" + "=" * 60)
    print(f"GRAD: {city_name}")
    print("=" * 60)

    print(f"Ukupno slika: {total_images}")
    print(f"Broj sekvenci: {len(sequence_counts)}")
    print(f"Slike bez sequence_id: {missing_sequence}")

    if not sequence_counts:
        print("Nema dostupnih sequence_id podataka.")
        return

    # Distribucija veličine sekvenci
    size_distribution = Counter(sequence_counts.values())

    print("\nDistribucija sekvenci:")

    for size in sorted(size_distribution):
        number_of_sequences = size_distribution[size]

        print(
            f"  {size} "
            f"{'slika' if size == 1 else 'slike'}: "
            f"{number_of_sequences} "
            f"{'sekvenca' if number_of_sequences == 1 else 'sekvence/sekvenci'}"
        )

    # Grupiranje većih sekvenci
    one_image = size_distribution.get(1, 0)
    two_images = size_distribution.get(2, 0)
    three_images = size_distribution.get(3, 0)

    four_or_more = sum(
        count
        for size, count in size_distribution.items()
        if size >= 4
    )

    print("\nSažetak:")

    print(f"  1 slika:  {one_image} sekvenci")
    print(f"  2 slike:  {two_images} sekvenci")
    print(f"  3 slike:  {three_images} sekvenci")
    print(f"  4+ slika: {four_or_more} sekvenci")

    counts = list(sequence_counts.values())

    print("\nStatistika:")

    print(
        f"  Prosječno slika po sekvenci: "
        f"{sum(counts) / len(counts):.2f}"
    )

    print(
        f"  Najveća sekvenca: "
        f"{max(counts)} slika"
    )

    print(
        f"  Najmanja sekvenca: "
        f"{min(counts)} slika"
    )


def main() -> None:
    if not DATASET_ROOT.exists():
        raise RuntimeError(
            f"Dataset mapa nije pronađena: "
            f"{DATASET_ROOT.resolve()}"
        )

    city_folders = sorted(
        folder
        for folder in DATASET_ROOT.iterdir()
        if folder.is_dir()
    )

    if not city_folders:
        print("Nema pronađenih gradova.")
        return

    print("=" * 60)
    print("ANALIZA MAPILLARY SEKVENCI")
    print("=" * 60)

    for city_folder in city_folders:
        analyze_city(city_folder)

    print("\n" + "=" * 60)
    print("ANALIZA ZAVRŠENA")
    print("=" * 60)


if __name__ == "__main__":
    main()
