"""
Fast Multi-threaded Downloader & Extractor for Oxford-IIIT Pet Dataset.
Downloads the official dataset archive from the FastAI S3 mirror and extracts into dataset_external/.
"""

import os
import sys
import time
import tarfile
from pathlib import Path
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


URL = "https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz"
DEST_DIR = Path("dataset_external")
TAR_PATH = DEST_DIR / "oxford-iiit-pet.tgz"
EXTRACT_DIR = DEST_DIR / "oxford-iiit-pet"
NUM_THREADS = 16


def download_chunk(url: str, start: int, end: int, chunk_idx: int, part_path: Path):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=30) as response, open(part_path, "wb") as f:
        while True:
            data = response.read(65536)
            if not data:
                break
            f.write(data)
    return chunk_idx


def download_multithreaded():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    if EXTRACT_DIR.exists() and (EXTRACT_DIR / "images").exists():
        img_count = len(list((EXTRACT_DIR / "images").glob("*.jpg")))
        if img_count >= 7300:
            print(f"Oxford-IIIT Pet already extracted: {img_count} images found.")
            return

    print(f"Connecting to S3 archive: {URL}...")
    req = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=15) as res:
        total_size = int(res.headers.get("Content-Length", 0))

    print(f"Total archive size: {total_size / (1024*1024):.2f} MB")
    chunk_size = total_size // NUM_THREADS
    parts = []

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        for i in range(NUM_THREADS):
            start = i * chunk_size
            end = total_size - 1 if i == NUM_THREADS - 1 else (i + 1) * chunk_size - 1
            part_path = DEST_DIR / f"part_{i}.tmp"
            parts.append(part_path)
            futures.append(executor.submit(download_chunk, URL, start, end, i, part_path))

        done_count = 0
        for fut in as_completed(futures):
            fut.result()
            done_count += 1
            sys.stdout.write(f"\rDownloading chunks: {done_count}/{NUM_THREADS} completed...")
            sys.stdout.flush()

    print(f"\nAll chunks downloaded in {time.time() - t0:.2f}s. Reassembling archive...")
    with open(TAR_PATH, "wb") as outfile:
        for part_path in parts:
            with open(part_path, "rb") as infile:
                while True:
                    buf = infile.read(1048576)
                    if not buf:
                        break
                    outfile.write(buf)
            part_path.unlink()

    print(f"Archive reassembled at {TAR_PATH}. Extracting...")
    t1 = time.time()
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        tar.extractall(path=DEST_DIR)

    # Clean up archive
    if TAR_PATH.exists():
        TAR_PATH.unlink()

    # The archive extracts into dataset_external/oxford-iiit-pet
    images_dir = EXTRACT_DIR / "images"
    extracted_imgs = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
    print(f"Extracted {extracted_imgs} images in {time.time() - t1:.2f}s.")
    print("Oxford-IIIT Pet dataset setup complete!")


if __name__ == "__main__":
    download_multithreaded()
