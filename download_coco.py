#!/usr/bin/env python3
"""
Download COCO 2017 Caption dataset.

This script downloads:
- Training images (train2017.zip)
- Validation images (val2017.zip)
- Annotations (annotations_trainval2017.zip)

Usage:
    python download_coco.py --data-dir /path/to/data
    python download_coco.py --data-dir /path/to/data --split train val  # Download specific splits
"""

import argparse
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# COCO 2017 dataset URLs
COCO_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "test_images": "http://images.cocodataset.org/zips/test2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

# File sizes for progress tracking (approximate, in bytes)
FILE_SIZES = {
    "train_images": 19_000_000_000,  # ~19GB
    "val_images": 1_000_000_000,  # ~1GB
    "test_images": 6_000_000_000,  # ~6GB
    "annotations": 241_000_000,  # ~241MB
}

# MD5 checksums (optional, for verification)
CHECKSUMS = {
    "train_images": None,  # Not available from official source
    "val_images": None,
    "test_images": None,
    "annotations": None,
}


class ProgressBar:
    """Simple progress bar for download progress."""

    def __init__(self, total_size: int, desc: str = "Downloading"):
        self.total_size = total_size
        self.desc = desc
        self.downloaded = 0
        self.last_percent = -1

    def update(self, chunk_size: int):
        """Update progress bar."""
        self.downloaded += chunk_size
        percent = int((self.downloaded / self.total_size) * 100) if self.total_size > 0 else 0

        if percent != self.last_percent:
            self.last_percent = percent
            # Simple progress display
            bar_length = 50
            filled = int(bar_length * percent / 100)
            bar = "=" * filled + "-" * (bar_length - filled)
            size_mb = self.downloaded / (1024 * 1024)
            total_mb = self.total_size / (1024 * 1024) if self.total_size > 0 else 0
            print(f"\r{self.desc}: [{bar}] {percent}% ({size_mb:.1f}MB / {total_mb:.1f}MB)", end="", flush=True)

    def finish(self):
        """Finish progress bar."""
        print()  # New line after progress bar


def download_file(url: str, filepath: Path, desc: str = "Downloading", expected_size: int | None = None):
    """
    Download a file with progress bar.

    Args:
        url: URL to download from
        filepath: Path to save the file
        desc: Description for progress bar
        expected_size: Expected file size in bytes (for progress bar)
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    if filepath.exists():
        file_size = filepath.stat().st_size
        if expected_size and file_size == expected_size:
            print(f"✓ File already exists: {filepath}")
            return True
        elif expected_size:
            print("⚠ File exists but size doesn't match. Re-downloading...")
            filepath.unlink()
        else:
            print(f"✓ File already exists: {filepath}")
            return True

    print(f"Downloading {filepath.name}...")
    print(f"URL: {url}")

    # Create progress bar
    progress = ProgressBar(expected_size or 0, desc=desc)

    def reporthook(blocknum, blocksize, totalsize):
        """Report download progress."""
        if totalsize > 0:
            progress.total_size = totalsize
        progress.update(blocksize)

    try:
        urlretrieve(url, str(filepath), reporthook=reporthook)
        progress.finish()
        print(f"✓ Downloaded: {filepath}")
        return True
    except Exception as e:
        progress.finish()
        print(f"✗ Error downloading {filepath.name}: {e}")
        if filepath.exists():
            filepath.unlink()  # Remove partial download
        return False


def extract_zip(zip_path: Path, extract_to: Path, desc: str = "Extracting", strip_prefix: str | None = None):
    """
    Extract a zip file.

    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
        desc: Description for extraction
        strip_prefix: If provided, strip this prefix from all paths in the zip
                     (e.g., "annotations/" to avoid nested directories)
    """
    print(f"{desc} {zip_path.name}...")
    extract_to.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Get total number of files for progress
            file_list = zip_ref.namelist()
            total_files = len(file_list)

            # Check if we need to strip a prefix
            if strip_prefix is None:
                # Auto-detect if all files have a common prefix (like "annotations/")
                if file_list:
                    first_path = file_list[0]
                    if "/" in first_path:
                        common_prefix = first_path.split("/")[0] + "/"
                        # Check if all files start with this prefix
                        if all(f.startswith(common_prefix) for f in file_list if f):
                            strip_prefix = common_prefix

            for idx, member in enumerate(file_list):
                # Skip directories
                if member.endswith("/"):
                    continue

                # Strip prefix if needed
                if strip_prefix and member.startswith(strip_prefix):
                    member_path = member[len(strip_prefix) :]
                else:
                    member_path = member

                # Extract to target location
                target_path = extract_to / member_path
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # Extract file
                with zip_ref.open(member) as source:
                    with open(target_path, "wb") as target:
                        target.write(source.read())

                if (idx + 1) % 100 == 0 or (idx + 1) == total_files:
                    percent = int((idx + 1) / total_files * 100)
                    print(f"\r{desc}: {percent}% ({idx + 1}/{total_files} files)", end="", flush=True)

        print()  # New line
        print(f"✓ Extracted: {zip_path.name}")
        return True
    except Exception as e:
        print(f"\n✗ Error extracting {zip_path.name}: {e}")
        return False


def verify_extraction(extract_dir: Path, expected_dirs: list[str]):
    """
    Verify that extraction was successful.

    Args:
        extract_dir: Directory where files were extracted
        expected_dirs: List of expected directory names
    """
    for expected_dir in expected_dirs:
        dir_path = extract_dir / expected_dir
        if not dir_path.exists():
            print(f"⚠ Warning: Expected directory not found: {dir_path}")
            return False
        if not dir_path.is_dir():
            print(f"⚠ Warning: Expected directory is not a directory: {dir_path}")
            return False
    return True


def download_coco_split(data_dir: Path, split: str, download_images: bool = True):
    """
    Download COCO dataset for a specific split.

    Args:
        data_dir: Root directory for COCO data
        split: Dataset split ('train', 'val', or 'test')
        download_images: Whether to download images (if False, only downloads annotations)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    downloads_dir = data_dir / "downloads"
    downloads_dir.mkdir(exist_ok=True)

    images_dir = data_dir / "images"

    # Download images if requested
    if download_images:
        image_key = f"{split}_images"
        if image_key in COCO_URLS:
            zip_name = f"{split}2017.zip"
            zip_path = downloads_dir / zip_name
            url = COCO_URLS[image_key]
            expected_size = FILE_SIZES.get(image_key)

            print(f"\n{'=' * 60}")
            print(f"Downloading {split} images...")
            print(f"{'=' * 60}")

            if download_file(url, zip_path, desc=f"Downloading {split} images", expected_size=expected_size):
                # Extract images
                extract_zip(zip_path, images_dir, desc=f"Extracting {split} images")
                # Verify extraction
                expected_image_dir = f"{split}2017"
                if verify_extraction(images_dir, [expected_image_dir]):
                    print(f"✓ {split} images extracted successfully")
                else:
                    print(f"⚠ Warning: {split} images extraction may have issues")

                # Optionally remove zip file to save space
                # zip_path.unlink()
                # print(f"✓ Removed zip file: {zip_path}")
            else:
                print(f"✗ Failed to download {split} images")
                return False
        else:
            print(f"⚠ Warning: No URL found for {split} images")

    return True


def download_coco_annotations(data_dir: Path):
    """
    Download COCO annotations.

    Args:
        data_dir: Root directory for COCO data
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    downloads_dir = data_dir / "downloads"
    downloads_dir.mkdir(exist_ok=True)

    annotations_dir = data_dir / "annotations"

    # Check if annotations already exist
    captions_file = annotations_dir / "captions_train2017.json"
    if captions_file.exists():
        print(f"✓ Annotations already exist: {captions_file}")
        return True

    # Download annotations
    zip_path = downloads_dir / "annotations_trainval2017.zip"
    url = COCO_URLS["annotations"]
    expected_size = FILE_SIZES.get("annotations")

    print(f"\n{'=' * 60}")
    print("Downloading annotations...")
    print(f"{'=' * 60}")

    if download_file(url, zip_path, desc="Downloading annotations", expected_size=expected_size):
        # Extract annotations (strip "annotations/" prefix if present in zip)
        extract_zip(zip_path, annotations_dir, desc="Extracting annotations", strip_prefix="annotations/")
        # Verify extraction
        expected_files = ["captions_train2017.json", "captions_val2017.json"]
        all_exist = all((annotations_dir / f).exists() for f in expected_files)
        if all_exist:
            print("✓ Annotations extracted successfully")
        else:
            print("⚠ Warning: Some annotation files may be missing")
            # Check if files are in nested directory
            nested_annotations = annotations_dir / "annotations"
            if nested_annotations.exists() and nested_annotations.is_dir():
                print("⚠ Found nested annotations directory. Moving files...")
                for file in nested_annotations.glob("*.json"):
                    shutil.move(str(file), str(annotations_dir / file.name))
                # Remove empty nested directory
                try:
                    nested_annotations.rmdir()
                    print("✓ Fixed nested directory structure")
                except OSError:
                    pass

        # Optionally remove zip file to save space
        # zip_path.unlink()
        # print(f"✓ Removed zip file: {zip_path}")
        return True
    else:
        print("✗ Failed to download annotations")
        return False


def main():
    """Main function to download COCO dataset."""
    parser = argparse.ArgumentParser(
        description="Download COCO 2017 Caption dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all splits (train, val, annotations)
  python download_coco.py --data-dir /path/to/coco

  # Download only validation split
  python download_coco.py --data-dir /path/to/coco --split val

  # Download train and val splits
  python download_coco.py --data-dir /path/to/coco --split train val

  # Download only annotations (no images)
  python download_coco.py --data-dir /path/to/coco --annotations-only
        """,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory to download COCO data (will create 'images' and 'annotations' subdirectories)",
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Dataset splits to download (default: all splits: train val test)",
    )
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Download only annotations (skip images)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove zip files after extraction to save space",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()

    print(f"\n{'=' * 60}")
    print("COCO 2017 Caption Dataset Downloader")
    print(f"{'=' * 60}")
    print(f"Data directory: {data_dir}")
    print(f"Splits to download: {args.split}")
    print(f"{'=' * 60}\n")

    # Download annotations (always needed)
    if not download_coco_annotations(data_dir):
        print("\n✗ Failed to download annotations. Exiting.")
        return 1

    # Download images for specified splits
    if not args.annotations_only:
        for split in args.split:
            if not download_coco_split(data_dir, split, download_images=True):
                print(f"\n✗ Failed to download {split} split. Continuing...")

    # Cleanup zip files if requested
    if args.cleanup:
        downloads_dir = data_dir / "downloads"
        if downloads_dir.exists():
            print(f"\nCleaning up zip files in {downloads_dir}...")
            for zip_file in downloads_dir.glob("*.zip"):
                zip_file.unlink()
                print(f"✓ Removed: {zip_file.name}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("Download Summary")
    print(f"{'=' * 60}")
    print(f"Data directory: {data_dir}")
    print(f"Images directory: {data_dir / 'images'}")
    print(f"Annotations directory: {data_dir / 'annotations'}")

    # Check what was downloaded
    images_dir = data_dir / "images"
    annotations_dir = data_dir / "annotations"

    if images_dir.exists():
        image_dirs = [d.name for d in images_dir.iterdir() if d.is_dir()]
        print(f"\nDownloaded image splits: {', '.join(image_dirs) if image_dirs else 'None'}")

    if annotations_dir.exists():
        annotation_files = [f.name for f in annotations_dir.glob("*.json")]
        print(f"Annotation files: {', '.join(annotation_files) if annotation_files else 'None'}")

    print(f"\n{'=' * 60}")
    print("Next steps:")
    print("1. Update your config file with:")
    print("   dataset:")
    print(f'     coco_root: "{data_dir / "images"}"')
    print(f'     coco_ann_file: "{data_dir / "annotations" / "captions_train2017.json"}"  # or captions_val2017.json')
    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    exit(main())
