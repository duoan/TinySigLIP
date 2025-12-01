"""
COCO Caption dataset loader for SigLIP training.
Directly uses cached embeddings and images from prepare_data.py.
No dependency on torchvision CocoCaptions.
"""

import json
import pickle
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_file_or_raise(file_path: Path, error_msg: str):
    """Load a file or raise FileNotFoundError with helpful message."""
    if not file_path.exists():
        raise FileNotFoundError(f"{error_msg}\nPlease run prepare_data.py first to generate the cache.")
    return file_path


class COCOCaptionDataset(Dataset):
    """
    COCO Caption dataset for SigLIP training.

    Directly loads data from cache created by prepare_data.py.
    Uses cached embeddings and loads images by image_path.
    """

    def __init__(
        self,
        dataset_path: str,
        processor=None,
        max_seq_len: int = 77,
        verbose: bool = True,
    ):
        """
        Args:
            dataset_path: Path to cached embeddings directory
                          (e.g., "data/coco/cache/model_name/dataset_size/split")
            processor: AutoProcessor for both student and teacher (same processor)
            max_seq_len: Maximum sequence length for text
            verbose: Whether to print verbose messages
        """
        self.dataset_path = Path(dataset_path)
        self.max_seq_len = max_seq_len
        self.verbose = verbose

        # Load metadata
        metadata_file = _load_file_or_raise(
            self.dataset_path / "metadata.json", f"Metadata file not found: {self.dataset_path / 'metadata.json'}"
        )
        with open(metadata_file, encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Build images directory path
        # Find cache directory parent (data/coco) and append images
        # Note: image_path in cache already includes split (e.g., "train/000000153656.jpg")
        current_path = self.dataset_path
        while current_path != current_path.parent:
            if current_path.name == "cache":
                self.images_dir = current_path.parent / "images"
                break
            current_path = current_path.parent
        else:
            raise ValueError(
                f"Could not infer images directory from dataset_path: {dataset_path}\n"
                "Expected format: data/coco/cache/model_name/dataset_size/split"
            )

        # Load index files
        index_file = _load_file_or_raise(
            self.dataset_path / "image_index.pt", f"Image index file not found: {self.dataset_path / 'image_index.pt'}"
        )
        self.image_index = torch.load(index_file, map_location="cpu")

        sample_indices_file = _load_file_or_raise(
            self.dataset_path / "sample_indices.pkl",
            f"Sample indices file not found: {self.dataset_path / 'sample_indices.pkl'}",
        )
        with open(sample_indices_file, "rb") as f:
            self.sample_indices = pickle.load(f)

        # Small in-memory cache for teacher batch files to avoid repeatedly
        # loading the same huge .pt file from disk for every sample.
        # This trades a bit of RAM for a big speed‑up on spinning disks / network FS.
        # Each entry is: batch_idx -> dict[local_idx, (image_id, image_path, image_emb, caption_data_list)]
        self._batch_cache: OrderedDict[int, dict] = OrderedDict()
        # Number of batch files to keep in memory per Dataset instance / DataLoader worker
        # 3–8 is usually a good trade‑off; each batch file size is controlled by images_per_batch in prepare_data.py
        self._max_cached_batches: int = 4

        # Validate processor
        if processor is None:
            raise ValueError("processor must be provided (use AutoProcessor)")
        self.processor = processor

        if verbose:
            print(f"✓ Loaded dataset from: {dataset_path}")
            print(f"  - Split: {self.metadata.get('split', 'unknown')}")
            print(f"  - Images: {self.metadata.get('num_images', 0)}")
            print(f"  - Captions: {self.metadata.get('total_captions', 0)}")
            print(f"  - Samples: {len(self.sample_indices)}")
            print(f"  - Image embed dim: {self.metadata.get('image_embed_dim', 0)}")
            print(f"  - Text embed dim: {self.metadata.get('text_embed_dim', 0)}")
            print(f"  - Images directory: {self.images_dir}")

    def _load_teacher_batch(self, batch_idx: int) -> dict:
        """Load a batch file from teacher cache directory with a small LRU in‑memory cache."""
        batch_file = self.dataset_path / f"batch_{batch_idx:06d}.pt"
        if not batch_file.exists():
            raise FileNotFoundError(f"Batch file not found: {batch_file}")

        # Fast path: served from in‑memory cache (per worker)
        if batch_idx in self._batch_cache:
            batch_data = self._batch_cache.pop(batch_idx)  # refresh LRU order
            self._batch_cache[batch_idx] = batch_data
            return batch_data

        # Slow path: load from disk once, then cache
        batch_data = torch.load(batch_file, map_location="cpu")
        self._batch_cache[batch_idx] = batch_data

        # Enforce simple LRU eviction to cap memory usage
        if len(self._batch_cache) > self._max_cached_batches:
            # pop the oldest inserted item
            self._batch_cache.popitem(last=False)

        return batch_data

    def _load_image_from_path(self, image_path: str) -> Image.Image:
        """Load image from file path (relative to images directory)."""
        full_path = self.images_dir / image_path
        if not full_path.exists():
            raise FileNotFoundError(f"Image file not found: {full_path}")
        return Image.open(full_path).convert("RGB")

    def _process_sample(self, image: Image.Image, caption: str):
        """Process a single image-caption pair."""
        # Clean and validate caption
        caption = str(caption).strip() or "a photo"

        # Preprocess using AutoProcessor
        inputs = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
        )

        # Get input_ids
        input_ids = inputs["input_ids"].squeeze(0)  # (seq_len,)

        # Get attention_mask if available, otherwise create it from input_ids
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"].squeeze(0)  # (seq_len,)
        else:
            # Create attention_mask: 1 for non-padding tokens, 0 for padding tokens
            # Assume pad_token_id is 0 if tokenizer has it, otherwise use 0 as default
            pad_token_id = 0
            if hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "pad_token_id"):
                if self.processor.tokenizer.pad_token_id is not None:
                    pad_token_id = self.processor.tokenizer.pad_token_id
            attention_mask = (input_ids != pad_token_id).long()

        return {
            "image": inputs["pixel_values"].squeeze(0),  # (C, H, W)
            "student_text_ids": input_ids,  # (seq_len,)
            "student_attention_mask": attention_mask,  # (seq_len,)
            "caption": caption,
        }

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.sample_indices)

    def __getitem__(self, idx):
        """Get a sample by index."""
        image_id, cap_idx = self.sample_indices[idx]

        # Get teacher embeddings from cache
        if image_id not in self.image_index:
            raise IndexError(f"Image ID {image_id} not found in teacher cache")

        batch_file_idx, local_idx = self.image_index[image_id]
        # Synchronously ensure the *requested* batch file is loaded.
        # Other batch files are handled asynchronously by the prefetch worker.
        batch_data = self._load_teacher_batch(batch_file_idx)

        if local_idx not in batch_data:
            raise IndexError(f"Local index {local_idx} not found in batch {batch_file_idx}")

        # Unpack: (image_id, image_path, image_emb, caption_data_list)
        cached_image_id, image_path, image_emb, caption_data_list = batch_data[local_idx]

        # Validate and get caption (fallback to first if index out of range)
        cap_idx = min(cap_idx, len(caption_data_list) - 1)
        _, caption_emb, caption_text = caption_data_list[cap_idx]

        # Load and process image
        image = self._load_image_from_path(image_path)
        result = self._process_sample(image, caption_text)

        # Add teacher embeddings and metadata
        result.update(
            {
                "teacher_image_embeds": image_emb,
                "teacher_text_embeds": caption_emb,
                "image_id": cached_image_id,
                "image_path": image_path,
            }
        )

        return result


def collate_coco_batch(batch):
    """
    Custom collate function for COCO dataset batches.

    Args:
        batch: List of items from COCOCaptionDataset

    Returns:
        dict with batched tensors
    """
    return {
        "student_images": torch.stack([item["image"] for item in batch]),
        "student_text_ids": torch.stack([item["student_text_ids"] for item in batch]),
        "student_attention_mask": torch.stack([item["student_attention_mask"] for item in batch]),
        "captions": [item["caption"] for item in batch],
        "teacher_image_embeds": torch.stack([item["teacher_image_embeds"] for item in batch]),
        "teacher_text_embeds": torch.stack([item["teacher_text_embeds"] for item in batch]),
        "image_ids": [item["image_id"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }
