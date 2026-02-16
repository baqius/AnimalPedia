"""
Animal Image Downloader v4.0
=============================
Downloads one real JPEG photo per animal from Wikipedia into ./animal_images/

FIXES in v4.0:
  - Uses Wikipedia's official THUMBNAIL endpoint (avoids 429 rate-limiting)
  - Exponential backoff on any HTTP error
  - Longer, randomised delays between requests
  - Skips SVG at every stage (URL check + Content-Type check)
  - Safe to re-run: already-saved files are skipped

Requirements:
    pip install requests Pillow

Usage:
    python download_animal_images.py
"""

import os
import time
import random
import requests
from PIL import Image
from io import BytesIO

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_FOLDER  = "animal_images"
THUMB_WIDTH    = 800          # thumbnail width requested from Wikimedia
JPEG_QUALITY   = 92
MIN_DELAY      = 1.0          # minimum seconds between animals
MAX_DELAY      = 2.5          # maximum seconds between animals (randomised)
MAX_RETRIES    = 4            # attempts per download before giving up
# ─────────────────────────────────────────────────────────────────────────────

ANIMALS = [
    "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar",
    "butterfly", "cat", "caterpillar", "chimpanzee", "cockroach", "cow",
    "coyote", "crab", "crow", "deer", "dog", "dolphin", "donkey", "dragonfly",
    "duck", "eagle", "elephant", "flamingo", "fly", "fox", "goat", "goldfish",
    "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog",
    "hippopotamus", "hornbill", "horse", "hummingbird", "hyena", "jellyfish",
    "kangaroo", "koala", "ladybugs", "leopard", "lion", "lizard", "lobster",
    "mosquito", "moth", "mouse", "octopus", "okapi", "orangutan", "otter",
    "owl", "ox", "oyster", "panda", "parrot", "pelecaniformes", "penguin",
    "pig", "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer",
    "rhinoceros", "sandpiper", "seahorse", "seal", "shark", "sheep", "snake",
    "sparrow", "squid", "squirrel", "starfish", "swan", "tiger", "turkey",
    "turtle", "whale", "wolf", "wombat", "woodpecker", "zebra",
]

SEARCH_OVERRIDES = {
    "ladybugs":       "ladybug insect",
    "fly":            "housefly insect Musca domestica",
    "bat":            "bat mammal chiroptera",
    "ox":             "domestic ox cattle",
    "possum":         "common brushtail possum",
    "pelecaniformes": "pelican bird",
    "moth":           "moth insect Lepidoptera",
    "rat":            "brown rat Rattus norvegicus",
    "crow":           "carrion crow Corvus corone",
    "cockroach":      "cockroach insect Blattodea",
    "mosquito":       "mosquito Culicidae insect",
    "dragonfly":      "dragonfly Odonata insect",
    "grasshopper":    "grasshopper Orthoptera insect",
    "beetle":         "beetle Coleoptera insect",
    "caterpillar":    "caterpillar larva Lepidoptera",
    "goldfish":       "goldfish Carassius auratus",
    "oyster":         "oyster Ostreidae mollusc",
    "squid":          "squid Teuthida cephalopod",
    "starfish":       "starfish Asteroidea",
    "lobster":        "lobster Homarus crustacean",
    "crab":           "crab Brachyura crustacean",
    "donkey":         "donkey Equus africanus asinus",
    "goose":          "Canada goose bird",
    "pigeon":         "rock pigeon Columba livia",
}

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

API_HEADERS = {
    "User-Agent": "AnimalImageDownloader/4.0 (educational; python-requests)"
}

# Browser headers for the actual image download
DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://en.wikipedia.org/",
}


def is_valid_url(url: str) -> bool:
    u = url.lower().split("?")[0]
    return any(u.endswith(e) for e in VALID_EXTENSIONS) and ".svg" not in u


def polite_sleep(base: float = None):
    """Sleep for a randomised polite interval."""
    delay = base if base else random.uniform(MIN_DELAY, MAX_DELAY)
    time.sleep(delay)


def wiki_api(params: dict) -> dict:
    """Call the Wikipedia API with error handling, return parsed JSON."""
    resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        headers=API_HEADERS, timeout=12, params={**params, "format": "json"}
    )
    resp.raise_for_status()
    return resp.json()


def get_page_title(search_term: str) -> str | None:
    try:
        data = wiki_api({"action": "query", "list": "search",
                         "srsearch": search_term, "srlimit": 3})
        results = data.get("query", {}).get("search", [])
        return results[0]["title"] if results else None
    except Exception:
        return None


def get_thumbnail_url(page_title: str) -> str | None:
    """
    Use Wikipedia's pageimages API with piprop=thumbnail to get a
    pre-rendered, size-limited thumbnail — the approach explicitly
    recommended by Wikimedia to avoid 429 errors.
    """
    try:
        data = wiki_api({
            "action": "query", "titles": page_title,
            "prop": "pageimages", "piprop": "thumbnail",
            "pithumbsize": THUMB_WIDTH,
        })
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            url = page.get("thumbnail", {}).get("source", "")
            if url and is_valid_url(url):
                return url
    except Exception:
        pass
    return None


def get_images_on_page(page_title: str) -> list[str]:
    """
    Fetch all image file titles listed on a page, resolve to thumbnail
    URLs via imageinfo, skip SVGs.
    """
    urls = []
    try:
        data = wiki_api({"action": "query", "titles": page_title,
                         "prop": "images", "imlimit": 15})
        pages = data.get("query", {}).get("pages", {})
        filenames = []
        for page in pages.values():
            for img in page.get("images", []):
                fname = img.get("title", "")
                if fname and is_valid_url(fname):
                    filenames.append(fname)

        if not filenames:
            return []

        # Resolve filenames → thumbnail URLs in one batch call
        data2 = wiki_api({
            "action": "query", "titles": "|".join(filenames[:10]),
            "prop": "imageinfo", "iiprop": "url",
            "iiurlwidth": THUMB_WIDTH,
        })
        for p in data2.get("query", {}).get("pages", {}).values():
            for ii in p.get("imageinfo", []):
                # Prefer thumburl (resized), fall back to url
                url = ii.get("thumburl") or ii.get("url", "")
                if url and is_valid_url(url):
                    urls.append(url)
    except Exception:
        pass
    return urls


def find_image_url(animal: str) -> str | None:
    """Three-stage search for a valid raster thumbnail URL."""
    term = SEARCH_OVERRIDES.get(animal, animal)

    # Stage 1: lead thumbnail from primary page
    title = get_page_title(term)
    if title:
        url = get_thumbnail_url(title)
        if url:
            return url
        # Stage 2: scan all images on the page
        imgs = get_images_on_page(title)
        if imgs:
            return imgs[0]

    # Stage 3: broaden the search
    title2 = get_page_title(term + " animal")
    if title2 and title2 != title:
        url = get_thumbnail_url(title2)
        if url:
            return url
        imgs2 = get_images_on_page(title2)
        if imgs2:
            return imgs2[0]

    return None


def download_and_save(url: str, filepath: str) -> bool:
    """Download a thumbnail URL and save as JPEG with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=DOWNLOAD_HEADERS, timeout=20, stream=True)

            # On 429, wait and retry with increasing delay
            if resp.status_code == 429:
                wait = 10 * (2 ** attempt)
                print(f"\n    ⏳ Rate limited (429) — waiting {wait}s ...", end=" ", flush=True)
                time.sleep(wait)
                continue

            # On 403, retry without Referer
            if resp.status_code == 403:
                h = {k: v for k, v in DOWNLOAD_HEADERS.items() if k != "Referer"}
                resp = requests.get(url, headers=h, timeout=20, stream=True)

            resp.raise_for_status()

            ct = resp.headers.get("Content-Type", "")
            if "svg" in ct or "html" in ct or "text" in ct:
                print(f"\n    ⚠  Skipping non-photo content ({ct})")
                return False

            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img.save(filepath, "JPEG", quality=JPEG_QUALITY, optimize=True)
            return True

        except requests.exceptions.HTTPError as e:
            wait = 5 * (2 ** attempt)
            if attempt < MAX_RETRIES - 1:
                print(f"\n    ↻ HTTP error ({e}) — retry in {wait}s ...", end=" ", flush=True)
                time.sleep(wait)
            else:
                print(f"\n    ✗ Giving up: {e}")

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(3)
            else:
                print(f"\n    ✗ Giving up: {e}")

    return False


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"\n🐾  Animal Image Downloader v4.0")
    print(f"{'─' * 54}")
    print(f"📁  Saving to    : ./{OUTPUT_FOLDER}/")
    print(f"🔢  Total animals: {len(ANIMALS)}")
    print(f"⏱   Delay        : {MIN_DELAY}–{MAX_DELAY}s between downloads\n")

    success, failed = 0, []

    for i, animal in enumerate(ANIMALS, 1):
        filepath = os.path.join(OUTPUT_FOLDER, f"{animal}.jpg")

        if os.path.exists(filepath):
            print(f"[{i:02d}/{len(ANIMALS)}] ⏭  {animal.capitalize()} — already exists")
            success += 1
            continue

        print(f"[{i:02d}/{len(ANIMALS)}] 🔍  {animal.capitalize()} ...", end=" ", flush=True)

        img_url = find_image_url(animal)

        if not img_url:
            print("✗  No image found")
            failed.append(animal)
            polite_sleep()
            continue

        saved = download_and_save(img_url, filepath)
        if saved:
            size_kb = os.path.getsize(filepath) // 1024
            print(f"✓  Saved ({size_kb} KB)")
            success += 1
        else:
            failed.append(animal)

        polite_sleep()

    print(f"\n{'─' * 54}")
    print(f"✅  Downloaded   : {success}/{len(ANIMALS)} images")
    if failed:
        print(f"❌  Failed ({len(failed)})   : {', '.join(failed)}")
    print(f"📁  Folder       : ./{OUTPUT_FOLDER}/")
    print(f"{'─' * 54}\n")


if __name__ == "__main__":
    main()