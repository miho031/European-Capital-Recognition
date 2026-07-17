import csv
import math
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


# ============================================================
# POSTAVKE
# ============================================================

CITY_NAME = "Zagreb"

# Središte Zagreba — za sada približno Trg bana Jelačića.
# Kasnije ćemo za sve gradove koristiti isti izvor koordinata.
CENTER_LAT = 45.8131
CENTER_LON = 15.9775

# Sve gradove prikupljamo unutar jednakog radijusa.
RADIUS_METERS = 1000

# Veličina jedne mrežne ćelije.
GRID_SPACING_METERS = 150

# Koliko kandidata Mapillary smije vratiti po ćeliji.
CANDIDATES_PER_CELL = 10

# Najviše slika iz iste Mapillary sekvence. (Sekvenca je niz fotografija snimljenih u nizu, npr. dok se vozite ulicom.)
MAX_IMAGES_PER_SEQUENCE = 3
# Povečano sa 1 na 3 zbog toga što večina čelija nije pronalazila odgovarajuču sliku, a svakako je svaka čelija ograničena na 1 sliku pa nebi trebalo dolaziti do preklapanja.

# Ukupan maksimalan broj spremljenih slika.
MAX_IMAGES = 100

# Pauza između API zahtjeva.
REQUEST_DELAY_SECONDS = 0.25

OUTPUT_ROOT = Path("dataset")
MAPILLARY_IMAGES_URL = "https://graph.mapillary.com/images"


# ============================================================
# POMOĆNE FUNKCIJE
# ============================================================

def meters_to_latitude_degrees(meters: float) -> float:
    """Približna pretvorba metara u stupnjeve geografske širine."""
    return meters / 111_320


def meters_to_longitude_degrees(
    meters: float,
    latitude: float,
) -> float:
    """Pretvorba metara u stupnjeve geografske dužine."""
    latitude_radians = math.radians(latitude)
    meters_per_degree = 111_320 * math.cos(latitude_radians)

    if abs(meters_per_degree) < 0.0001:
        raise ValueError("Pretvorba nije moguća blizu polova.")

    return meters / meters_per_degree


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Vraća udaljenost između dviju GPS koordinata u metrima."""

    earth_radius = 6_371_000

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2) ** 2
    )

    return 2 * earth_radius * math.asin(math.sqrt(value))


def generate_grid_points(
    center_lat: float,
    center_lon: float,
    radius_meters: int,
    spacing_meters: int,
) -> list[tuple[float, float]]:
    """
    Generira pravilnu mrežu točaka, ali zadržava samo one koje
    se nalaze unutar kružnog područja oko centra.
    """

    points: list[tuple[float, float]] = []

    offset_x = -radius_meters

    while offset_x <= radius_meters:
        offset_y = -radius_meters

        while offset_y <= radius_meters:
            distance_from_center = math.sqrt(
                offset_x**2 + offset_y**2
            )

            if distance_from_center <= radius_meters:
                latitude = (
                    center_lat
                    + meters_to_latitude_degrees(offset_y)
                )

                longitude = (
                    center_lon
                    + meters_to_longitude_degrees(
                        offset_x,
                        center_lat,
                    )
                )

                points.append((latitude, longitude))

            offset_y += spacing_meters

        offset_x += spacing_meters

    # Prvo obrađujemo točke bliže centru.
    points.sort(
        key=lambda point: haversine_distance(
            center_lat,
            center_lon,
            point[0],
            point[1],
        )
    )

    return points


def create_cell_bbox(
    latitude: float,
    longitude: float,
    cell_size_meters: int,
) -> str:
    """
    Stvara mali bbox oko mrežne točke.

    Format:
    west,south,east,north
    """

    half_size = cell_size_meters / 2

    latitude_delta = meters_to_latitude_degrees(half_size)
    longitude_delta = meters_to_longitude_degrees(
        half_size,
        latitude,
    )

    west = longitude - longitude_delta
    south = latitude - latitude_delta
    east = longitude + longitude_delta
    north = latitude + latitude_delta

    return f"{west},{south},{east},{north}"


def extract_coordinates(
    image: dict[str, Any],
) -> tuple[float, float] | None:
    """
    Mapillary computed_geometry koristi GeoJSON redoslijed:
    longitude, latitude.
    """

    geometry = image.get("computed_geometry")

    if not geometry:
        return None

    coordinates = geometry.get("coordinates")

    if not coordinates or len(coordinates) < 2:
        return None

    longitude = float(coordinates[0])
    latitude = float(coordinates[1])

    return latitude, longitude


def get_sequence_id(image: dict[str, Any]) -> str | None:
    """
    Polje sequence može biti ID, tekst ili objekt,
    ovisno o odgovoru API-ja.
    """

    sequence = image.get("sequence")

    if sequence is None:
        return None

    if isinstance(sequence, dict):
        sequence_id = sequence.get("id")
        return str(sequence_id) if sequence_id else None

    return str(sequence)


def fetch_candidates(
    access_token: str,
    bbox: str,
) -> list[dict[str, Any]]:
    """Dohvaća moguće fotografije iz jedne male ćelije."""

    params = {
        "access_token": access_token,
        "fields": (
            "id,"
            "computed_geometry,"
            "sequence,"
            "captured_at,"
            "camera_type,"
            "thumb_1024_url"
        ),
        "bbox": bbox,
        "limit": CANDIDATES_PER_CELL,
    }

    response = requests.get(
        MAPILLARY_IMAGES_URL,
        params=params,
        timeout=30,
    )

    if not response.ok:
        print(
            f"API greška {response.status_code}: "
            f"{response.text[:300]}"
        )
        return []

    return response.json().get("data", [])


def choose_best_candidate(
    candidates: list[dict[str, Any]],
    grid_latitude: float,
    grid_longitude: float,
    used_image_ids: set[str],
    sequence_counts: dict[str, int],
) -> dict[str, Any] | None:
    """
    Odabire neiskorištenu sliku najbližu središtu mrežne ćelije.
    """

    ranked_candidates = []

    for image in candidates:
        camera_type = image.get("camera_type")
        # Uklanjamo 360° panoramske i fisheye fotografije jer su često iskrivljene i neupotrebljive.
        if camera_type != "perspective":
            continue

        image_id = str(image.get("id", ""))

        if not image_id or image_id in used_image_ids:
            continue

        image_url = image.get("thumb_1024_url")

        if not image_url:
            continue

        coordinates = extract_coordinates(image)

        if coordinates is None:
            continue

        image_latitude, image_longitude = coordinates
        sequence_id = get_sequence_id(image)

        if sequence_id:
            current_count = sequence_counts.get(sequence_id, 0)

            if current_count >= MAX_IMAGES_PER_SEQUENCE:
                continue

        distance = haversine_distance(
            grid_latitude,
            grid_longitude,
            image_latitude,
            image_longitude,
        )

        ranked_candidates.append((distance, image))

    if not ranked_candidates:
        return None

    ranked_candidates.sort(key=lambda item: item[0])

    return ranked_candidates[0][1]


def download_image(
    image_url: str,
    destination: Path,
) -> bool:
    """Preuzima jednu fotografiju."""

    try:
        response = requests.get(
            image_url,
            timeout=60,
        )
        response.raise_for_status()

        destination.write_bytes(response.content)
        return True

    except requests.RequestException as error:
        print(f"Neuspjelo preuzimanje: {error}")
        return False


# ============================================================
# GLAVNI PROGRAM
# ============================================================

def main() -> None:
    load_dotenv()

    access_token = os.getenv("MAPILLARY_ACCESS_TOKEN")

    if not access_token:
        raise RuntimeError(
            "MAPILLARY_ACCESS_TOKEN nije pronađen u .env datoteci."
        )

    city_folder = OUTPUT_ROOT / CITY_NAME
    images_folder = city_folder / "images"

    images_folder.mkdir(parents=True, exist_ok=True)

    metadata_path = city_folder / "metadata.csv"

    grid_points = generate_grid_points(
        center_lat=CENTER_LAT,
        center_lon=CENTER_LON,
        radius_meters=RADIUS_METERS,
        spacing_meters=GRID_SPACING_METERS,
    )

    print(f"Grad: {CITY_NAME}")
    print(f"Broj mrežnih točaka: {len(grid_points)}")
    print(f"Radijus: {RADIUS_METERS} m")
    print()

    used_image_ids: set[str] = set()
    sequence_counts: dict[str, int] = {}
    metadata_rows: list[dict[str, Any]] = []

    for grid_index, (grid_lat, grid_lon) in enumerate(
        grid_points,
        start=1,
    ):
        if len(used_image_ids) >= MAX_IMAGES:
            break

        bbox = create_cell_bbox(
            latitude=grid_lat,
            longitude=grid_lon,
            cell_size_meters=GRID_SPACING_METERS,
        )

        print(
            f"[{grid_index}/{len(grid_points)}] "
            f"Provjeravam ćeliju..."
        )

        candidates = fetch_candidates(
            access_token=access_token,
            bbox=bbox,
        )

        selected = choose_best_candidate(
            candidates=candidates,
            grid_latitude=grid_lat,
            grid_longitude=grid_lon,
            used_image_ids=used_image_ids,
            sequence_counts=sequence_counts,
        )

        if selected is None:
            print("  Nema odgovarajuće slike.")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        image_id = str(selected["id"])
        image_url = selected["thumb_1024_url"]
        sequence_id = get_sequence_id(selected)
        coordinates = extract_coordinates(selected)

        if coordinates is None:
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        image_latitude, image_longitude = coordinates

        center_distance = haversine_distance(
            CENTER_LAT,
            CENTER_LON,
            image_latitude,
            image_longitude,
        )

        # Dodatna zaštita: slika mora biti unutar zadanog kruga.
        if center_distance > RADIUS_METERS:
            print("  Slika je izvan zadanog radijusa.")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        filename = f"{CITY_NAME.lower()}_{len(used_image_ids) + 1:04d}.jpg"
        destination = images_folder / filename

        success = download_image(
            image_url=image_url,
            destination=destination,
        )

        if not success:
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        used_image_ids.add(image_id)

        if sequence_id:
            sequence_counts[sequence_id] = (
                sequence_counts.get(sequence_id, 0) + 1
            )

        metadata_rows.append(
            {
                "filename": filename,
                "city": CITY_NAME,
                "image_id": image_id,
                "sequence_id": sequence_id or "",
                "latitude": image_latitude,
                "longitude": image_longitude,
                "distance_from_center_m": round(
                    center_distance,
                    2,
                ),
                "grid_latitude": grid_lat,
                "grid_longitude": grid_lon,
                "camera_type": selected.get("camera_type", ""),
                "captured_at": selected.get(
                    "captured_at",
                    "",
                ),
            }
        )

        print(f"  Spremljeno: {filename}")

        time.sleep(REQUEST_DELAY_SECONDS)

    with metadata_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        fieldnames = [
            "filename",
            "city",
            "image_id",
            "sequence_id",
            "latitude",
            "longitude",
            "distance_from_center_m",
            "grid_latitude",
            "grid_longitude",
            "captured_at",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(metadata_rows)

    print()
    print("Prikupljanje završeno.")
    print(f"Spremljeno slika: {len(metadata_rows)}")
    print(f"Slike: {images_folder}")
    print(f"Metapodaci: {metadata_path}")


if __name__ == "__main__":
    main()
