#!/usr/bin/env python3
"""
Unified data preparation script for TinySigLIP.

This script handles all data preparation steps:
1. Load COCO 2017 dataset from HuggingFace (phiyodr/coco2017)
2. Download images (only the ones needed)
3. Extract and cache teacher model embeddings
4. Support small datasets for local development (e.g., 1K samples)
5. Automatic multi-GPU support for faster embedding extraction

Usage:
    python prepare_data.py --data-dir data/coco
    python prepare_data.py --data-dir data/coco --max-samples 1000  # Small dataset for local dev
    python prepare_data.py --data-dir data/coco --skip-download  # Skip download if data exists
    python prepare_data.py --data-dir data/coco --skip-embeddings  # Skip embedding extraction
    python prepare_data.py --data-dir data/coco --num-workers 16  # Use more parallel workers for faster download

Note: For faster downloads, install requests library:
    pip install requests

The script uses parallel downloads (default: 8 workers) which significantly speeds up
image downloading compared to sequential downloads.

Multi-GPU Support:
    The script automatically detects and uses all available CUDA GPUs for embedding extraction.
    Images are distributed across GPUs using round-robin scheduling for optimal load balancing.
    If multiple GPUs are detected, the teacher model will be replicated to each GPU.

"""

import argparse
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import torch
from PIL import Image


def get_device():
    """Get the appropriate device for computation."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_available_devices():
    """
    Get all available devices for computation.

    Returns:
        list: List of available devices. For CUDA, returns all available GPUs.
              For MPS, returns single MPS device. For CPU, returns single CPU device.
    """
    devices = []
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        print(f"Detected {num_gpus} CUDA GPU(s): {[str(d) for d in devices]}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices = [torch.device("mps")]
        print("Detected MPS device")
    else:
        devices = [torch.device("cpu")]
        print("Using CPU device")
    return devices


def get_device_info(device):
    """
    Get device information for logging purposes.

    Args:
        device: Device to get info for

    Returns:
        str: Device information string
    """
    if str(device).startswith("cuda") and device.index is not None:
        try:
            import torch

            if torch.cuda.is_available():
                gpu_id = device.index
                total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
                total_memory_gb = total_memory / (1024**3)
                return f"GPU {gpu_id} ({total_memory_gb:.2f} GB)"
        except Exception:
            pass
    return str(device)


def download_image(url: str, filepath: Path, timeout: int = 10) -> bool:
    """Download a single image from URL using requests (faster than urlretrieve)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    if filepath.exists():
        return True

    try:
        try:
            import requests

            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except ImportError:
            # Fallback to urlretrieve if requests not available
            urlretrieve(url, str(filepath))
            return True
    except Exception:
        # Final fallback
        try:
            urlretrieve(url, str(filepath))
            return True
        except Exception:
            return False


def load_coco_from_hf(data_dir: Path, split: str = "train", max_samples: int | None = None):
    """
    Load COCO dataset from HuggingFace.

    Args:
        data_dir: Root directory for COCO data
        split: Dataset split ('train' or 'validation')
        max_samples: Maximum number of samples to load (None for full dataset)

    Returns:
        Dataset from HuggingFace
    """
    try:
        from datasets import load_dataset

        print("Loading COCO dataset from HuggingFace: phiyodr/coco2017")
        print(f"  Split: {split}")
        if max_samples:
            print(f"  Max samples: {max_samples} (small dataset for local dev)")

        # Load dataset from HuggingFace
        hf_dataset: Any = load_dataset("phiyodr/coco2017", split=split)

        # Limit dataset size if requested
        if max_samples and hasattr(hf_dataset, "__len__"):
            dataset_len = len(hf_dataset)  # type: ignore[arg-type]
            if max_samples < dataset_len:
                print(f"  Limiting to {max_samples} samples (from {dataset_len} total)")
                if hasattr(hf_dataset, "select"):
                    hf_dataset = hf_dataset.select(range(max_samples))  # type: ignore[union-attr]

        dataset_len = len(hf_dataset) if hasattr(hf_dataset, "__len__") else "unknown"  # type: ignore[arg-type]
        print(f"✓ Loaded {dataset_len} samples from HuggingFace")
        return hf_dataset

    except ImportError:
        print("✗ Error: datasets library not installed")
        print("  Install with: pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def download_image_parallel(args_tuple: tuple[str, Path]) -> tuple[bool, str]:
    """Download a single image (for parallel execution)."""
    url, image_path = args_tuple
    success = download_image(url, image_path)
    return success, image_path.name


def download_coco_images(dataset: Any, data_dir: Path, split: str, num_workers: int = 8):
    """
    Download COCO images from URLs in the dataset using parallel downloads.

    Args:
        dataset: HuggingFace dataset
        data_dir: Root directory for COCO data
        split: Dataset split ('train' or 'validation')
        num_workers: Number of parallel download workers (default: 8)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tqdm import tqdm

    # Map split names: 'validation' -> 'val', others -> split
    split_dir_name = "val" if split == "validation" else split
    images_dir = data_dir / "images" / split_dir_name
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading images to: {images_dir}")
    dataset_len = len(dataset) if hasattr(dataset, "__len__") else "unknown"  # type: ignore[arg-type]
    print(f"Total images to download: {dataset_len}")
    print(f"Using {num_workers} parallel workers for faster download")

    # Prepare download tasks
    download_tasks = []
    for example in dataset:
        # Get image file name (e.g., "train/000000391895.jpg")
        file_name = example["file_name"]
        image_filename = file_name.split("/")[-1]  # Extract just filename
        image_path = images_dir / image_filename

        # Skip if already downloaded
        if image_path.exists():
            continue

        # Get image URL (prefer coco_url as it's more reliable)
        coco_url = example.get("coco_url")
        flickr_url = example.get("flickr_url")

        # Try to download from coco_url first, then flickr_url
        url = coco_url if coco_url else flickr_url
        if not url:
            continue

        download_tasks.append((url, image_path))

    if not download_tasks:
        total_count = len(dataset) if hasattr(dataset, "__len__") else 0  # type: ignore[arg-type]
        print("✓ All images already downloaded!")
        return 0, total_count, 0

    skipped_count = len(dataset) - len(download_tasks) if hasattr(dataset, "__len__") else 0  # type: ignore[arg-type]
    print(f"Downloading {len(download_tasks)} images (skipping {skipped_count} already downloaded)...")

    # Track progress
    downloaded = 0
    failed = 0
    skipped = len(dataset) - len(download_tasks) if hasattr(dataset, "__len__") else 0  # type: ignore[arg-type]

    # Parallel download with progress bar
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(download_image_parallel, task): task for task in download_tasks}

        # Process completed downloads with progress bar
        with tqdm(total=len(download_tasks), desc="Downloading images") as pbar:
            for future in as_completed(future_to_task):
                try:
                    success, filename = future.result()
                    if success:
                        downloaded += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
                pbar.update(1)

    print("\n✓ Image download completed!")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")

    return downloaded, skipped, failed


def get_dataset_size_name(max_samples: int | None) -> str:
    """
    Get dataset size name based on max_samples.

    Args:
        max_samples: Maximum number of samples (None for full dataset)

    Returns:
        Size name: "tiny" (<=1k), "medium" (<=10k), or "large" (>10k or None)
    """
    if max_samples is None:
        return "large"
    elif max_samples <= 1000:
        return "tiny"
    elif max_samples <= 10000:
        return "medium"
    else:
        return "large"


def extract_teacher_embeddings(
    teacher_model,
    teacher_processor,
    hf_dataset,  # HuggingFace dataset
    devices,  # List of devices or single device
    teacher_model_name: str,
    split: str,
    max_samples: int | None = None,
    batch_size: int = 32,
    cache_dir: Path | str | None = None,
    num_workers: int = 4,
):
    """
    Extract teacher embeddings grouped by image_id.

    Each image is only extracted once, with all its captions.
    Save format in batch files: {local_idx: (image_id, image_path, image_emb, list[(caption_id, caption_emb, caption)])}

    This avoids duplicate image extraction and makes data loading much faster.

    Args:
        teacher_model: Teacher model instance (already loaded)
        teacher_processor: Teacher processor instance
        hf_dataset: HuggingFace dataset
        devices: List of devices or single device to run on. If list, will distribute work across devices.
        teacher_model_name: Name/identifier of the teacher model (e.g., "google/siglip2-base-patch16-224")
        split: Dataset split ('train', 'val', etc.)
        max_samples: Maximum number of samples used (None for full dataset)
        batch_size: Batch size parameter (kept for compatibility, not actively used since COCO has max 4 captions)
        cache_dir: Base cache directory
        num_workers: Number of data loading workers (not used in this function)
    """
    """
    Extract teacher embeddings grouped by image_id.

    Each image is only extracted once, with all its captions.
    Save format in batch files: {local_idx: (image_id, image_path, image_emb, list[(caption_id, caption_emb, caption)])}

    This avoids duplicate image extraction and makes data loading much faster.

    Args:
        teacher_model_name: Name/identifier of the teacher model (e.g., "google/siglip2-base-patch16-224")
                           Used to create a separate cache directory for each model.
        split: Dataset split ('train', 'val', etc.) - used to separate cache by split.
        max_samples: Maximum number of samples used (None for full dataset).
                    Used to create a separate cache directory for different dataset sizes.
        num_workers: Number of data loading workers. Set to 0 to disable multiprocessing
                    (useful if encountering pickle errors).
    """
    import re

    from tqdm import tqdm

    # Normalize devices to list
    if not isinstance(devices, list):
        devices = [devices]

    num_devices = len(devices)
    use_multi_gpu = num_devices > 1 and all(str(d).startswith("cuda") for d in devices)

    if use_multi_gpu:
        print(f"Using {num_devices} GPUs for parallel processing")
        # Replicate model to all GPUs
        teacher_models = {}
        for device in devices:
            teacher_models[device] = teacher_model.to(device)
            teacher_models[device].eval()
            print(f"  {get_device_info(device)}")
    else:
        # Single device: use the first device
        device = devices[0]
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        print(f"Using single device: {get_device_info(device)}")

    # Sanitize model name for filesystem (replace special chars with _)
    safe_model_name = re.sub(r"[^\w\-_]", "_", teacher_model_name).strip("_")

    # Get dataset size name (tiny/medium/large)
    dataset_size_name = get_dataset_size_name(max_samples)

    # Normalize split name (val -> val, validation -> val, etc.)
    split_name = "val" if split in ["val", "validation"] else split

    # Create cache directory: base_cache_dir / model_name / dataset_size / split
    base_cache_dir = Path(cache_dir) if cache_dir else Path("data/coco/cache")
    cache_dir = base_cache_dir / safe_model_name / dataset_size_name / split_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Teacher model: {teacher_model_name}")
    print(f"Dataset size: {dataset_size_name} (max_samples={max_samples})")
    print(f"Split: {split_name}")
    print(f"Cache directory: {cache_dir}")

    # Get max_position_embeddings from loaded teacher model
    max_position_embeddings = teacher_model.config.text_config.max_position_embeddings
    print(f"Model max_position_embeddings: {max_position_embeddings}")

    # Map split names: 'validation' -> 'val', others -> split
    split_dir_name = "val" if split == "validation" else split

    # Infer images directory from cache_dir convention: data/coco/cache/... -> data/coco/images
    cache_dir_path = Path(cache_dir)
    current_path = cache_dir_path
    while current_path != current_path.parent:
        if current_path.name == "cache":
            images_dir = current_path.parent / "images" / split_dir_name
            break
        current_path = current_path.parent
    else:
        raise ValueError(
            f"Could not infer images directory from cache_dir: {cache_dir}\n"
            "Expected format: data/coco/cache/model_name/dataset_size/split"
        )

    print(f"Extracting teacher embeddings for {len(hf_dataset)} images...")

    # Determine embedding dimensions by processing a dummy sample first
    print("Determining embedding dimensions...")
    with torch.no_grad():
        sample_example = hf_dataset[0]
        # Load image from file
        file_name = sample_example["file_name"]
        image_filename = file_name.split("/")[-1]
        image_path = images_dir / image_filename
        sample_image = Image.open(image_path).convert("RGB")

        # Get captions
        sample_captions = sample_example.get("captions", [])
        if isinstance(sample_captions, str):
            sample_captions = [sample_captions]
        if len(sample_captions) == 0:
            sample_captions = ["a photo"]

        # Process sample image and caption
        sample_inputs = teacher_processor(
            text=[sample_captions[0]],
            images=[sample_image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        # Use first device for sample
        sample_device = devices[0]
        sample_images = sample_inputs["pixel_values"].to(sample_device)
        sample_text_ids = sample_inputs["input_ids"].to(sample_device)

        if use_multi_gpu:
            sample_model = teacher_models[sample_device]
        else:
            sample_model = teacher_model

        sample_outputs = sample_model(
            pixel_values=sample_images,
            input_ids=sample_text_ids,
            output_hidden_states=True,
        )
        image_embed_dim = sample_outputs.image_embeds.shape[1]
        text_embed_dim = sample_outputs.text_embeds.shape[1]
        del sample_image, sample_inputs, sample_images, sample_text_ids, sample_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"  Image embed dim: {image_embed_dim}")
    print(f"  Text embed dim: {text_embed_dim}")

    # Process images grouped by image_id, save in batches
    print("Extracting embeddings (grouped by image, each image extracted only once)...")
    print("Saving in separate batch files for efficient memory usage...")

    # Configuration: save every N images to a batch file
    images_per_batch = 100  # Save every 100 images to a separate file

    # Processing batch size for GPU (larger batch = better GPU utilization)
    processing_batch_size = batch_size if batch_size > 0 else 32

    # Storage for current batch
    current_batch_data = {}
    batch_file_counter = 0
    caption_id_counter = 0
    total_captions = 0
    total_images = 0

    # Index file: maps image_id to (batch_file_idx, local_idx_in_batch)
    image_index = {}  # image_id -> (batch_file_idx, local_idx)
    # Sample indices: maps sample_idx -> (image_id, caption_idx) for expanding multiple captions per image
    sample_indices = []

    def save_batch(batch_data: dict, batch_idx: int):
        """Save a batch of image data to disk."""
        batch_file = cache_dir / f"batch_{batch_idx:06d}.pt"
        torch.save(batch_data, batch_file)
        return batch_file

    # Track processed image_ids to avoid duplicates
    processed_image_ids = set()

    with torch.no_grad():
        # Process images in batches for better GPU utilization
        image_batch_items = []  # List of (idx, example, image_id, image_path, image, captions, device_idx)

        def process_image_batch(batch_items):
            """Process a batch of images together for better GPU utilization."""
            nonlocal caption_id_counter, total_captions, current_batch_data, image_index

            if not batch_items:
                return

            # Group by device for multi-GPU
            device_batches = {}
            for item in batch_items:
                idx, example, image_id, image_path, image, captions, device_idx = item
                if device_idx not in device_batches:
                    device_batches[device_idx] = []
                device_batches[device_idx].append(item)

            # Process each device batch
            for device_idx, items in device_batches.items():
                current_device = devices[device_idx] if use_multi_gpu else devices[0]
                current_model = teacher_models[current_device] if use_multi_gpu else teacher_model

                # Extract images and prepare batch
                batch_images = [item[4] for item in items]  # Extract images
                batch_image_ids = [item[2] for item in items]  # Extract image_ids
                batch_image_paths = [item[3] for item in items]  # Extract image_paths
                batch_captions_list = [item[5] for item in items]  # Extract captions lists

                # Batch process images
                image_tensors = teacher_processor.image_processor(batch_images, return_tensors="pt")["pixel_values"].to(
                    current_device
                )

                # Get image features in batch
                image_embs = current_model.get_image_features(pixel_values=image_tensors).detach().cpu()
                # (batch_size, D_proj)

                # Process captions for all images
                all_captions = []
                caption_offsets = [0]  # Track where each image's captions start in all_captions

                for image_id, captions in zip(batch_image_ids, batch_captions_list, strict=True):
                    # Build sample_indices for all captions
                    for cap_idx in range(len(captions)):
                        sample_indices.append((image_id, cap_idx))
                        all_captions.append(captions[cap_idx])
                    caption_offsets.append(len(all_captions))

                # Process captions in batch if any
                if all_captions:
                    # Ensure captions are strings
                    captions_clean = [str(cap) if not isinstance(cap, str) else cap for cap in all_captions]

                    # Batch process all captions
                    max_length = max_position_embeddings
                    caption_text_ids = teacher_processor.tokenizer(
                        captions_clean,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                    )["input_ids"].to(current_device)

                    # Get text features in batch
                    caption_embs = current_model.get_text_features(input_ids=caption_text_ids).detach().cpu()
                    # (total_captions, D_proj)

                    del caption_text_ids
                else:
                    caption_embs = None

                # Group captions back to images and store
                for img_idx, (image_id, image_path, image_emb, captions) in enumerate(
                    zip(batch_image_ids, batch_image_paths, image_embs, batch_captions_list, strict=True)
                ):
                    caption_data_list = []
                    caption_start = caption_offsets[img_idx]
                    caption_end = caption_offsets[img_idx + 1]

                    if caption_embs is not None and caption_start < caption_end:
                        for cap_idx, caption in enumerate(captions):
                            caption_emb = caption_embs[caption_start + cap_idx]
                            caption_data_list.append((caption_id_counter, caption_emb, caption))
                            caption_id_counter += 1
                            total_captions += 1

                    # Store in current batch
                    local_idx = len(current_batch_data)
                    current_batch_data[local_idx] = (image_id, image_path, image_emb, caption_data_list)
                    image_index[image_id] = (batch_file_counter, local_idx)

                # Clean up
                del image_tensors, image_embs
                if caption_embs is not None:
                    del caption_embs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Collect and process images in batches
        for idx in tqdm(range(len(hf_dataset)), desc="Processing images"):
            try:
                example = hf_dataset[idx]

                # Get image_id
                image_id = example["image_id"]
                if isinstance(image_id, list):
                    image_id = image_id[0] if image_id else idx

                # Convert to Python int safely
                try:
                    if hasattr(image_id, "item"):
                        image_id = int(image_id.item())
                    elif isinstance(image_id, str):
                        image_id = int(image_id)
                    elif not isinstance(image_id, int):
                        image_id = int(image_id)
                except (ValueError, OverflowError, TypeError) as e:
                    print(
                        f"Warning: Failed to convert image_id {image_id} (type: {type(image_id)}): {e}. "
                        f"Using index {idx} instead."
                    )
                    image_id = idx

                # Skip if already processed
                if image_id in processed_image_ids:
                    continue
                processed_image_ids.add(image_id)

                # Load image from file
                file_name = example["file_name"]
                image_filename = file_name.split("/")[-1]
                image_path_file = images_dir / image_filename
                if not image_path_file.exists():
                    print(f"Warning: Image file not found: {image_path_file}. Skipping.")
                    continue
                image = Image.open(image_path_file).convert("RGB")

                # Get captions
                captions = example.get("captions", [])
                if isinstance(captions, str):
                    captions = [captions]
                if len(captions) == 0:
                    captions = ["a photo"]

                # Build relative path
                image_path = f"{split_name}/{image_filename}"

                # Select device for this image (round-robin for multi-GPU)
                device_idx = idx % num_devices if use_multi_gpu else 0

                # Add to batch
                image_batch_items.append((idx, example, image_id, image_path, image, captions, device_idx))

                # Process batch when it reaches the processing batch size
                if len(image_batch_items) >= processing_batch_size:
                    process_image_batch(image_batch_items)
                    image_batch_items = []  # Clear batch

                    # Save batch when it reaches the threshold
                    if len(current_batch_data) >= images_per_batch:
                        batch_file = save_batch(current_batch_data, batch_file_counter)
                        print(
                            f"  Saved batch {batch_file_counter} with {len(current_batch_data)} images "
                            f"to {batch_file.name}"
                        )
                        total_images += len(current_batch_data)
                        current_batch_data = {}  # Clear for next batch
                        batch_file_counter += 1

            except Exception as e:
                import traceback

                error_msg = str(e)
                if "int too big" in error_msg.lower() or "overflow" in error_msg.lower():
                    print(f"\n{'=' * 60}")
                    print(f"ERROR: Failed to process image {idx}: {e}")
                    print(f"{'=' * 60}")
                    traceback.print_exc()
                    print(f"{'=' * 60}\n")
                else:
                    print(f"Warning: Failed to process image {idx}: {e}. Skipping.")
                continue

        # Process remaining images
        if image_batch_items:
            process_image_batch(image_batch_items)

        # Save remaining images in the last batch
        if len(current_batch_data) > 0:
            batch_file = save_batch(current_batch_data, batch_file_counter)
            print(
                f"  Saved final batch {batch_file_counter} with {len(current_batch_data)} images to {batch_file.name}"
            )
            total_images += len(current_batch_data)
            batch_file_counter += 1

    # Save index file: maps image_id to (batch_file_idx, local_idx)
    print("Creating index file...")
    index_file = cache_dir / "image_index.pt"
    torch.save(image_index, index_file)
    print(f"  Saved index file: {index_file.name} ({len(image_index)} images indexed)")

    # Save sample_indices: maps sample_idx -> (image_idx, caption_idx)
    print("Saving sample indices...")
    import pickle

    sample_indices_file = cache_dir / "sample_indices.pkl"
    with open(sample_indices_file, "wb") as f:
        pickle.dump(sample_indices, f)
    print(f"  Saved sample indices: {sample_indices_file.name} ({len(sample_indices)} samples)")

    # Save metadata as JSON (easier to read and inspect)
    import json

    metadata = {
        "teacher_model_name": teacher_model_name,
        "split": split_name,
        "max_samples": max_samples,
        "dataset_size_name": dataset_size_name,
        "num_images": total_images,
        "total_captions": total_captions,
        "num_batches": batch_file_counter,
        "images_per_batch": images_per_batch,
        "image_embed_dim": image_embed_dim,
        "text_embed_dim": text_embed_dim,
        "cache_dir": str(cache_dir),
        "index_file": str(index_file),
        "sample_indices_file": str(sample_indices_file),
    }
    metadata_file = cache_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("✓ Cache saved successfully!")
    print(f"  - Total images: {total_images}")
    print(f"  - Total captions: {total_captions}")
    print(f"  - Number of batch files: {batch_file_counter}")
    print(f"  - Images per batch: {images_per_batch}")
    print(f"  - Image embed dim: {image_embed_dim}")
    print(f"  - Text embed dim: {text_embed_dim}")
    print(f"  - Cache directory: {cache_dir}")
    print(f"  - Index file: {index_file}")
    print(f"  - Sample indices file: {sample_indices_file}")
    print(f"  - Metadata file: {metadata_file}")

    return {
        "num_images": total_images,
        "total_captions": total_captions,
        "num_batches": batch_file_counter,
        "image_embed_dim": image_embed_dim,
        "text_embed_dim": text_embed_dim,
    }


def main():
    """Main function to prepare all data."""
    parser = argparse.ArgumentParser(
        description="Prepare all data for TinySigLIP training using HuggingFace datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare full dataset
  python prepare_data.py --data-dir data/coco

  # Prepare small dataset for local development (5K samples)
  python prepare_data.py --data-dir data/coco --max-samples 5000

  # Skip downloading images (if already downloaded)
  python prepare_data.py --data-dir data/coco --skip-download

  # Skip embedding extraction (only prepare data)
  python prepare_data.py --data-dir data/coco --skip-embeddings
        """,
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "val"],
        default="train",
        help="Dataset split to prepare (default: train). Use 'val' or 'validation' for validation set.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for small dataset, e.g., 5000 for local dev)",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="google/siglip2-base-patch16-224",
        help="HuggingFace model name for teacher model",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to save cached teacher embeddings (default: {data-dir}/cache)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size for preprocessing",
    )
    parser.add_argument(
        "--text-seq-len",
        type=int,
        default=64,
        help="Maximum text sequence length",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading images (assume they already exist)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip extracting teacher embeddings",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers for image download (default: 8)",
    )
    parser.add_argument(
        "--download-zip",
        action="store_true",
        help="Download images from COCO official zip files instead (much faster for full dataset)",
    )

    args = parser.parse_args()

    # Convert to Path objects
    cache_dir = Path(args.cache_dir).resolve()

    # Infer data_dir from cache_dir convention: data/coco/cache/... -> data/coco
    current_path = cache_dir
    while current_path != current_path.parent:
        if current_path.name == "cache":
            data_dir = current_path.parent
            break
        current_path = current_path.parent
    else:
        raise ValueError(f"Could not infer data_dir from cache_dir: {cache_dir}\nExpected format: data/coco/cache/...")

    print("\n" + "=" * 70)
    print("TinySigLIP Data Preparation (using HuggingFace datasets)")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Split: {args.split}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples} (small dataset for local dev)")
    else:
        print("Max samples: Full dataset")
    print(f"Teacher model: {args.teacher_model}")
    print(f"Cache directory: {cache_dir}")
    print("=" * 70 + "\n")

    # Step 1: Load dataset from HuggingFace
    print("\n" + "=" * 70)
    print("Step 1: Loading COCO Dataset from HuggingFace")
    print("=" * 70)

    # Map split names: 'val' -> 'validation' for HuggingFace
    hf_split = "validation" if args.split in ["val", "validation"] else "train"

    dataset = load_coco_from_hf(data_dir, split=hf_split, max_samples=args.max_samples)

    # Step 2: Download images
    if not args.skip_download:
        print("\n" + "=" * 70)
        print("Step 2: Downloading COCO Images")
        print("=" * 70)

        download_coco_images(dataset, data_dir, args.split, num_workers=args.num_workers)
    else:
        print("\n" + "=" * 70)
        print("Step 2: Skipping Image Download")
        print("=" * 70)

    # Step 4: Extract teacher embeddings
    if not args.skip_embeddings:
        print("\n" + "=" * 70)
        print("Step 4: Extracting Teacher Embeddings")
        print("=" * 70)

        # Check if images exist
        split_dir_name = "val" if args.split == "validation" else args.split
        images_dir = data_dir / "images" / split_dir_name

        if not images_dir.exists() or len(list(images_dir.glob("*.jpg"))) == 0:
            print(f"✗ Error: COCO images directory not found or empty: {images_dir}")
            print("   Please download images first or remove --skip-download flag")
            return 1

        # Load teacher model and processor
        try:
            from transformers import AutoModel, AutoProcessor

            # Auto-detect available devices
            devices = get_available_devices()

            # Load model (will be replicated to multiple devices in extract_teacher_embeddings if multi-GPU)
            print(f"Loading teacher model: {args.teacher_model}")
            teacher_model = AutoModel.from_pretrained(args.teacher_model)
            # Don't move to device here - will be handled in extract_teacher_embeddings for multi-GPU support

            print(f"Loading teacher processor: {args.teacher_model}")
            teacher_processor = AutoProcessor.from_pretrained(args.teacher_model)
            print("✓ Teacher model and processor loaded")
        except Exception as e:
            print(f"✗ Error loading teacher model: {e}")
            print("   Please check that transformers library is installed: pip install transformers")
            return 1

        # No need to create COCOCaptionDataset, we use HuggingFace dataset directly

        # Extract and cache teacher embeddings
        try:
            extract_teacher_embeddings(
                teacher_model=teacher_model,
                teacher_processor=teacher_processor,
                hf_dataset=dataset,  # HuggingFace dataset
                devices=devices,  # List of devices for multi-GPU support
                teacher_model_name=args.teacher_model,
                split=args.split,
                max_samples=args.max_samples,
                batch_size=args.batch_size,
                cache_dir=str(cache_dir),
                num_workers=0,  # Set to 0 to avoid multiprocessing issues, can be increased if needed
            )
        except Exception as e:
            print(f"✗ Error extracting embeddings: {e}")
            import traceback

            traceback.print_exc()
            return 1
    else:
        print("\n" + "=" * 70)
        print("Step 4: Skipping Embedding Extraction")
        print("=" * 70)

    # Print summary
    print("\n" + "=" * 70)
    print("Data Preparation Summary")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Images directory: {data_dir / 'images'}")

    # Check what was prepared
    split_dir_name = "val" if args.split == "validation" else args.split
    images_dir = data_dir / "images" / split_dir_name

    if images_dir.exists():
        num_images = len(list(images_dir.glob("*.jpg")))
        print(f"\nDownloaded images: {num_images}")

    # Check if embeddings were cached (check for metadata.json)
    metadata_file_path = cache_dir / "metadata.json"
    if metadata_file_path.exists():
        print(f"\n✓ Teacher embeddings cache found at: {cache_dir}")
    else:
        print(f"\n⚠ Teacher embeddings cache not found at: {cache_dir}")

    print("\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)
    print("1. Update config/config.yaml with:")
    print("   dataset:")
    print(f'     dataset_path: "{cache_dir}"  # Cache directory path')
    print("\n2. Start training:")
    print("   python train.py")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
