from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError


IMAGE_FOLDER = Path("dataset/raw/Zagreb")
DUPLICATES_FOLDER = IMAGE_FOLDER / "duplicates"

# Manji broj znači strože uspoređivanje.
# 0 = potpuno isti perceptual hash
# 5–8 = vrlo slične fotografije
HASH_THRESHOLD = 10
# rezultati se nisu uvijek pokazali doslijedni, pa je prag postavljen na 10.
# uvedene su dodatne provjere kako bi se smanjilo ponavljanje sličnih slika pa ovo služi kao posljednja linija obrane. Ako se pojave slične slike, one će biti premještene u mapu "duplicates".


def find_similar_images(folder: Path) -> None:
    DUPLICATES_FOLDER.mkdir(exist_ok=True)

    accepted_images: list[tuple[Path, imagehash.ImageHash]] = []

    image_paths = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                current_hash = imagehash.phash(image.convert("RGB"))
        except (UnidentifiedImageError, OSError) as error:
            print(f"Ne mogu otvoriti {image_path.name}: {error}")
            continue

        duplicate_of = None

        for accepted_path, accepted_hash in accepted_images:
            distance = current_hash - accepted_hash

            if distance <= HASH_THRESHOLD:
                duplicate_of = accepted_path
                break

        if duplicate_of:
            destination = DUPLICATES_FOLDER / image_path.name
            image_path.rename(destination)

            print(
                f"Slična slika: {image_path.name} "
                f"→ sliči na {duplicate_of.name}"
            )
        else:
            accepted_images.append((image_path, current_hash))
            print(f"Zadržana: {image_path.name}")

    print()
    print(f"Zadržano slika: {len(accepted_images)}")
    print(f"Slične slike spremljene su u: {DUPLICATES_FOLDER}")


if __name__ == "__main__":
    find_similar_images(IMAGE_FOLDER)
