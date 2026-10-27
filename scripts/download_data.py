"""Download the published UCI daily data without modifying its original bytes."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DAY_SHA256 = "a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be"


def main() -> None:
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        files = {name: archive.read(name) for name in ("day.csv", "Readme.txt")}
    if hashlib.sha256(files["day.csv"]).hexdigest() != DAY_SHA256:
        raise ValueError("Source day.csv has changed; review it before updating the pinned checksum.")

    destination = ROOT / "data" / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (destination / name).write_bytes(content)

    metadata = {
        "dataset": "Bike Sharing",
        "creator": "Hadi Fanaee-T",
        "repository": "UCI Machine Learning Repository",
        "dataset_url": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
        "doi": "https://doi.org/10.24432/C5W894",
        "download_url": SOURCE_URL,
        "license": "CC BY 4.0 (as stated on the UCI dataset page)",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "files_sha256": {
            name: hashlib.sha256(content).hexdigest() for name, content in files.items()
        },
    }
    (ROOT / "data" / "source.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Downloaded and verified {destination / 'day.csv'}")


if __name__ == "__main__":
    main()
