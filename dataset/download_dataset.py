"""
UIEB Dataset Downloader & Validator
Automates downloading larjeck/uieb-dataset-raw via kagglehub/Kaggle API into dataset/uieb_raw
"""
import os
import sys
import shutil
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

DATASET_SLUG = "larjeck/uieb-dataset-raw"
TARGET_RAW_DIR = config.UIEB_RAW_DIR


def download_uieb_raw():
    """Download the UIEB raw dataset using kagglehub."""
    print(f"\n=======================================================")
    print(f" Downloading UIEB Raw Dataset: {DATASET_SLUG}")
    print(f" Target Directory: {TARGET_RAW_DIR}")
    print(f"=======================================================\n")

    TARGET_RAW_DIR.parent.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub
        print("[+] Fetching dataset via kagglehub...")
        cache_path = Path(kagglehub.dataset_download(DATASET_SLUG))
        print(f"[+] Download complete at cache: {cache_path}")

        # Copy/link into target workspace directory
        print(f"[+] Linking files to {TARGET_RAW_DIR}...")
        for item in cache_path.iterdir():
            target = TARGET_RAW_DIR.parent / item.name
            if item.is_dir():
                if not target.exists():
                    shutil.copytree(item, target)
            else:
                if not target.exists():
                    shutil.copy2(item, target)

        num_images = len(list(TARGET_RAW_DIR.glob("*.png")) + list(TARGET_RAW_DIR.glob("*.jpg")))
        print(f"[+] Successfully ready: {num_images} real underwater images in {TARGET_RAW_DIR}")
        return True
    except Exception as e:
        print(f"[-] Error downloading UIEB dataset: {e}")
        return False


if __name__ == "__main__":
    download_uieb_raw()
