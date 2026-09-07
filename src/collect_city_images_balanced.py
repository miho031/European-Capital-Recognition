import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


# ============================================================
# POSTAVKE
# ============================================================

# Jednako prostorno područje za svaki grad.
RADIUS_METERS = 1000

# Cilj je prikupiti nešto više slika nego što će ostati nakon filtriranja.
TARGET_IMAGES_PER_CITY = 350

# Ako grad ostane ispod ove vrijednosti, ispisuje se upozorenje.
MIN_DESIRED_IMAGES_PER_CITY = 300

# Više prolaza omogućuje rjeđe pokrivenim gradovima da se približe cilju,
# bez povećavanja ukupnog gradskog radijusa od 1000 m.
# (grid_spacing_m, search_radius_m)
COLLECTION_PASSES = [
    (75, 35),
    (60, 50),
    (50, 70),
]

# Kod API 500 greške privremeno smanjujemo radijus pretrage za istu točku.
SEARCH_RADIUS_500_RETRIES = 4
SEARCH_RADIUS_500_FACTOR = 0.5
MIN_SEARCH_RADIUS_METERS = 5

# Broj kandidata koje Mapillary smije vratiti po mrežnoj točki.
CANDIDATES_PER_CELL = 20

# Ograničenje radi veće raznolikosti scena.
MAX_IMAGES_PER_SEQUENCE = 3

# Pauza između API zahtjeva.
REQUEST_DELAY_SECONDS = 0.25

# Reproducibilno miješanje mrežnih točaka.
RANDOM_SEED = 42

# Sigurnosna opcija. Ako je False, skripta neće pregaziti postojeće slike grada.
RESET_CITY_BEFORE_COLLECTION = False

OUTPUT_ROOT = Path("dataset/raw_new")
MAPILLARY_IMAGES_URL = "https://graph.mapillary.com/images"


# ============================================================
# POMOĆNE FUNKCIJE
# ============================================================


def load_cities() -> list[dict[str, Any]]:
    """Učitava popis gradova iz config/cities.json."""
    with open("config/cities.json", "r", encoding="utf-8") as file:
        return json.load(file)


def meters_to_latitude_degrees(meters: float) -> float:
    return meters / 111_320


def meters_to_longitude_degrees(meters: float, latitude: float) -> float:
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
    """Generira mrežu točaka unutar kružnog područja."""

    points: list[tuple[float, float]] = []

    offset_x = -radius_meters

    while offset_x <= radius_meters:
        offset_y = -radius_meters

        while offset_y <= radius_meters:
            distance_from_center = math.sqrt(offset_x**2 + offset_y**2)

            if distance_from_center <= radius_meters:
                latitude = center_lat + meters_to_latitude_degrees(offset_y)
                longitude = center_lon + meters_to_longitude_degrees(
                    offset_x,
                    center_lat,
                )
                points.append((latitude, longitude))

            offset_y += spacing_meters

        offset_x += spacing_meters

    return points


def create_cell_bbox(
    latitude: float,
    longitude: float,
    search_radius_meters: int,
) -> str:
    """Stvara bbox oko jedne mrežne točke: west,south,east,north."""

    latitude_delta = meters_to_latitude_degrees(search_radius_meters)
    longitude_delta = meters_to_longitude_degrees(
        search_radius_meters,
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
    """Mapillary computed_geometry koristi GeoJSON redoslijed lon, lat."""

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
    sequence = image.get("sequence")

    if sequence is None:
        return None

    if isinstance(sequence, dict):
        sequence_id = sequence.get("id")
        return str(sequence_id) if sequence_id else None

    return str(sequence)


def fetch_candidates(
    access_token: str,
    latitude: float,
    longitude: float,
    initial_search_radius_meters: int,
) -> list[dict[str, Any]] | None:
    """
    Dohvaća kandidate za jednu mrežnu točku.

    None znači da je ćelija završila trajnom API 500 greškom.
    [] znači da nema uporabljivih kandidata ili se dogodila druga greška.
    """

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
        "limit": CANDIDATES_PER_CELL,
    }

    max_retries = 3
    search_radius = initial_search_radius_meters

    for radius_attempt in range(SEARCH_RADIUS_500_RETRIES):
        bbox = create_cell_bbox(
            latitude=latitude,
            longitude=longitude,
            search_radius_meters=search_radius,
        )
        params["bbox"] = bbox

        got_500 = False

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    MAPILLARY_IMAGES_URL,
                    params=params,
                    timeout=45,
                )

                if response.ok:
                    return response.json().get("data", [])

                if response.status_code == 500:
                    got_500 = True
                    print(
                        f"API 500 za radijus {search_radius} m: "
                        f"{response.text[:200]}"
                    )
                    break

                print(
                    f"API greška {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return []

            except (
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ConnectionError,
            ) as error:
                print(
                    f"Pokušaj {attempt + 1}/{max_retries} nije uspio: "
                    f"{error}"
                )

                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    return []

        if not got_500:
            return []

        next_radius = max(
            MIN_SEARCH_RADIUS_METERS,
            int(search_radius * SEARCH_RADIUS_500_FACTOR),
        )

        if next_radius == search_radius:
            break

        search_radius = next_radius

        if radius_attempt < SEARCH_RADIUS_500_RETRIES - 1:
            print(
                f"Smanjujem radijus na {search_radius} m i pokušavam ponovno."
            )
            time.sleep(REQUEST_DELAY_SECONDS)

    return None


def choose_best_candidate(
    candidates: list[dict[str, Any]],
    grid_latitude: float,
    grid_longitude: float,
    used_image_ids: set[str],
    sequence_counts: dict[str, int],
) -> dict[str, Any] | None:
    """
    Odabire kandidata tako da prvo preferira slabije zastupljene sekvence,
    a zatim fotografiju najbližu mrežnoj točki.
    """

    ranked_candidates: list[tuple[int, float, dict[str, Any]]] = []

    for image in candidates:
        if image.get("camera_type") != "perspective":
            continue

        image_id = str(image.get("id", ""))

        if not image_id or image_id in used_image_ids:
            continue

        if not image.get("thumb_1024_url"):
            continue

        coordinates = extract_coordinates(image)

        if coordinates is None:
            continue

        image_latitude, image_longitude = coordinates
        sequence_id = get_sequence_id(image)
        sequence_count = 0

        if sequence_id:
            sequence_count = sequence_counts.get(sequence_id, 0)

            if sequence_count >= MAX_IMAGES_PER_SEQUENCE:
                continue

        distance = haversine_distance(
            grid_latitude,
            grid_longitude,
            image_latitude,
            image_longitude,
        )

        ranked_candidates.append((sequence_count, distance, image))

    if not ranked_candidates:
        return None

    ranked_candidates.sort(key=lambda item: (item[0], item[1]))

    return ranked_candidates[0][2]


def download_image(image_url: str, destination: Path) -> bool:
    try:
        response = requests.get(image_url, timeout=60)
        response.raise_for_status()
        destination.write_bytes(response.content)
        return True

    except requests.RequestException as error:
        print(f"Neuspjelo preuzimanje: {error}")
        return False


def write_metadata(
    metadata_path: Path,
    metadata_rows: list[dict[str, Any]],
) -> None:
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
        "camera_type",
        "captured_at",
        "collection_pass",
        "grid_spacing_m",
        "search_radius_m",
    ]

    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)


def prepare_city_output(city_folder: Path) -> Path:
    images_folder = city_folder / "images"

    if city_folder.exists() and any(city_folder.iterdir()):
        if RESET_CITY_BEFORE_COLLECTION:
            shutil.rmtree(city_folder)
        else:
            raise RuntimeError(
                f"Folder za grad već sadrži podatke: {city_folder}. "
                "Za svježe standardizirano prikupljanje obriši/premjesti "
                "postojeći folder ili postavi RESET_CITY_BEFORE_COLLECTION = True."
            )

    images_folder.mkdir(parents=True, exist_ok=True)
    return images_folder


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

    cities = load_cities()

    for city in cities:
        city_name = city["city"]
        center_lat = city["latitude"]
        center_lon = city["longitude"]

        print("\n" + "=" * 70)
        print(f"Prikupljanje grada: {city_name}")
        print("=" * 70)

        city_folder = OUTPUT_ROOT / city_name
        images_folder = prepare_city_output(city_folder)
        metadata_path = city_folder / "metadata.csv"

        used_image_ids: set[str] = set()
        sequence_counts: dict[str, int] = {}
        metadata_rows: list[dict[str, Any]] = []

        no_candidates = 0
        api_errors = 0

        for pass_number, (grid_spacing, search_radius) in enumerate(
            COLLECTION_PASSES,
            start=1,
        ):
            if len(used_image_ids) >= TARGET_IMAGES_PER_CITY:
                break

            grid_points = generate_grid_points(
                center_lat,
                center_lon,
                radius_meters=RADIUS_METERS,
                spacing_meters=grid_spacing,
            )

            # Ne obrađujemo uvijek centar prvi. Točke se miješaju na
            # reproduktivan način radi ravnomjernije prostorne zastupljenosti.
            random_generator = random.Random(
                f"{RANDOM_SEED}:{city_name}:{pass_number}"
            )
            random_generator.shuffle(grid_points)

            print(
                f"\nProlaz {pass_number}/{len(COLLECTION_PASSES)} | "
                f"grid={grid_spacing} m | search={search_radius} m | "
                f"točaka={len(grid_points)} | "
                f"trenutno slika={len(used_image_ids)}"
            )

            for grid_index, (grid_lat, grid_lon) in enumerate(
                grid_points,
                start=1,
            ):
                if len(used_image_ids) >= TARGET_IMAGES_PER_CITY:
                    break

                candidates = fetch_candidates(
                    access_token=access_token,
                    latitude=grid_lat,
                    longitude=grid_lon,
                    initial_search_radius_meters=search_radius,
                )

                if candidates is None:
                    api_errors += 1
                    continue

                selected = choose_best_candidate(
                    candidates=candidates,
                    grid_latitude=grid_lat,
                    grid_longitude=grid_lon,
                    used_image_ids=used_image_ids,
                    sequence_counts=sequence_counts,
                )

                if selected is None:
                    no_candidates += 1
                    time.sleep(REQUEST_DELAY_SECONDS)
                    continue

                image_id = str(selected["id"])
                image_url = selected["thumb_1024_url"]
                sequence_id = get_sequence_id(selected)
                coordinates = extract_coordinates(selected)

                if coordinates is None:
                    no_candidates += 1
                    continue

                image_latitude, image_longitude = coordinates

                center_distance = haversine_distance(
                    center_lat,
                    center_lon,
                    image_latitude,
                    image_longitude,
                )

                # Svi gradovi i svi prolazi ostaju unutar istih 1000 m.
                if center_distance > RADIUS_METERS:
                    continue

                filename = (
                    f"{city_name.lower()}_{len(used_image_ids) + 1:04d}.jpg"
                )
                destination = images_folder / filename

                if not download_image(image_url, destination):
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
                        "city": city_name,
                        "image_id": image_id,
                        "sequence_id": sequence_id or "",
                        "latitude": image_latitude,
                        "longitude": image_longitude,
                        "distance_from_center_m": round(center_distance, 2),
                        "grid_latitude": grid_lat,
                        "grid_longitude": grid_lon,
                        "camera_type": selected.get("camera_type", ""),
                        "captured_at": selected.get("captured_at", ""),
                        "collection_pass": pass_number,
                        "grid_spacing_m": grid_spacing,
                        "search_radius_m": search_radius,
                    }
                )

                if len(metadata_rows) % 25 == 0:
                    print(
                        f"  Spremljeno {len(metadata_rows)}/"
                        f"{TARGET_IMAGES_PER_CITY} slika..."
                    )

                time.sleep(REQUEST_DELAY_SECONDS)

            # Čuvamo metadata i nakon svakog prolaza, tako da se rezultat
            # ne izgubi ako kasnije dođe do prekida.
            write_metadata(metadata_path, metadata_rows)

            print(
                f"Završen prolaz {pass_number}: "
                f"ukupno {len(metadata_rows)} slika."
            )

        write_metadata(metadata_path, metadata_rows)

        print("\n===== Rezultat grada =====")
        print(f"Grad: {city_name}")
        print(f"Spremljeno slika: {len(metadata_rows)}")
        print(f"Različitih sekvenci: {len(sequence_counts)}")
        print(f"Bez kandidata: {no_candidates}")
        print(f"API 500 ćelije: {api_errors}")

        if len(metadata_rows) >= TARGET_IMAGES_PER_CITY:
            print("Cilj prikupljanja dosegnut.")
        elif len(metadata_rows) >= MIN_DESIRED_IMAGES_PER_CITY:
            print(
                "Cilj od 350 nije dosegnut, ali grad ima najmanje "
                f"{MIN_DESIRED_IMAGES_PER_CITY} fotografija."
            )
        else:
            print(
                "UPOZORENJE: grad je ostao ispod minimalno željenog broja "
                f"od {MIN_DESIRED_IMAGES_PER_CITY} fotografija."
            )

        time.sleep(0.3)


if __name__ == "__main__":
    main()
