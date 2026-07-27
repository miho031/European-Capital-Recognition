import requests
import os
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN")

if not ACCESS_TOKEN:
    raise RuntimeError("MAPILLARY_ACCESS_TOKEN nije pronađen u .env datoteci.")

CITY_NAME = "Zagreb"
OUTPUT_DIR = f"dataset/{CITY_NAME}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

response = requests.get(
    "https://graph.mapillary.com/images",
    params={
        "access_token": ACCESS_TOKEN,
        "fields": "id,thumb_1024_url",
        "bbox": "15.9740,45.8105,15.9810,45.8155",
        "limit": 10,
    },
    timeout=30,
)

print("Status:", response.status_code)

if not response.ok:
    print(response.text)
    raise SystemExit

images = response.json().get("data", [])

print("Pronađeno slika:", len(images))

for index, image in enumerate(images, start=1):
    image_url = image.get("thumb_1024_url")

    if not image_url:
        continue

    image_response = requests.get(image_url, timeout=30)

    if not image_response.ok:
        print(f"Neuspjelo preuzimanje slike {index}")
        continue

    file_path = os.path.join(
        OUTPUT_DIR,
        f"{CITY_NAME.lower()}_{index:03d}.jpg"
    )

    with open(file_path, "wb") as file:
        file.write(image_response.content)

    print("Spremljeno:", file_path)
