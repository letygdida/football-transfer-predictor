"""
download_data.py
----------------
Downloads the Transfermarkt dataset CSV files from the publicly hosted R2 bucket
maintained by dcaribou/transfermarkt-datasets (https://github.com/dcaribou/transfermarkt-datasets).

Data is scraped from Transfermarkt and published under a community data-sharing ethos.
Please respect Transfermarkt's terms of service and do not use this data commercially.

Files downloaded:
  - players.csv.gz         -> data/raw/players.csv
  - appearances.csv.gz     -> data/raw/appearances.csv
  - clubs.csv.gz           -> data/raw/clubs.csv
  - competitions.csv.gz    -> data/raw/competitions.csv

Run:
  python src/download_data.py
"""

import os
import sys
import gzip
import shutil
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"

FILES = {
    "players.csv.gz":      "players.csv",
    "appearances.csv.gz":  "appearances.csv",
    "clubs.csv.gz":        "clubs.csv",
    "competitions.csv.gz": "competitions.csv",
}

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _progress_hook(block_count: int, block_size: int, total_size: int) -> None:
    """Simple download progress indicator."""
    if total_size > 0:
        downloaded = block_count * block_size
        pct = min(downloaded / total_size * 100, 100)
        mb_done = downloaded / 1_048_576
        mb_total = total_size / 1_048_576
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)
    else:
        print(f"\r  Downloaded {block_count * block_size / 1_048_576:.1f} MB", end="", flush=True)


def download_and_decompress(gz_filename: str, csv_filename: str, raw_dir: Path) -> bool:
    """
    Download a .csv.gz file and decompress it to raw_dir/csv_filename.
    Returns True on success, False if the output already exists (skip).
    """
    out_path = raw_dir / csv_filename

    if out_path.exists() and out_path.stat().st_size > 1_000:
        print(f"  ✓ {csv_filename} already exists — skipping download.")
        return True

    url = f"{BASE_URL}/{gz_filename}"
    gz_path = raw_dir / gz_filename

    print(f"\n→ Downloading {gz_filename} ...")
    try:
        urllib.request.urlretrieve(url, gz_path, reporthook=_progress_hook)
        print()  # newline after progress bar
    except urllib.error.URLError as exc:
        print(f"\n  ✗ Failed to download {gz_filename}: {exc}")
        return False

    print(f"  Decompressing → {csv_filename} ...")
    try:
        with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()  # remove the compressed file to save space
        size_mb = out_path.stat().st_size / 1_048_576
        print(f"  ✓ {csv_filename}  ({size_mb:.1f} MB)")
        return True
    except Exception as exc:
        print(f"  ✗ Decompression failed for {gz_filename}: {exc}")
        # Clean up partial files
        if gz_path.exists():
            gz_path.unlink()
        if out_path.exists():
            out_path.unlink()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Transfermarkt Dataset Downloader")
    print("  Source: github.com/dcaribou/transfermarkt-datasets")
    print("=" * 60)

    success_count = 0
    for gz_name, csv_name in FILES.items():
        ok = download_and_decompress(gz_name, csv_name, RAW_DIR)
        if ok:
            success_count += 1

    print()
    print(f"{'=' * 60}")
    print(f"  Done: {success_count}/{len(FILES)} files ready in data/raw/")

    if success_count < len(FILES):
        print()
        print("  Some files failed to download.  Manual download instructions:")
        print(f"  1. Visit: {BASE_URL}")
        print("  2. Download each .csv.gz file listed above.")
        print(f"  3. Decompress them and place the .csv files in: {RAW_DIR}")
        sys.exit(1)

    print()
    print("  Next step: python src/preprocess.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
