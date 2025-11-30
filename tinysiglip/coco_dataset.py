"""
COCO Caption 2017 dataset loader for SigLIP training.
Uses IterableDataset for streaming and regular Dataset for non-streaming.
Expands multiple captions per image.
"""

import hashlib
import os
import pickle
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset, IterableDataset

from tinysiglip.processor import TinySiglipProcessor


class COCOCaptionIterableDataset(IterableDataset):
    """
    COCO Caption 2017 dataset for SigLIP training (streaming mode).

    Uses IterableDataset for streaming data loading.
    """

    def __init__(
        self,
        split: str = "val",
        image_size: int = 224,
        student_processor: TinySiglipProcessor | None = None,
        student_tokenizer=None,  # Deprecated: use student_processor instead
        teacher_processor=None,
        max_seq_len: int = 77,
        cache_dir: str | None = None,
        use_augmentation: bool = False,
    ):
        """
        Args:
            split: Dataset split ('test', 'val', or 'train' which maps to 'test').
                   test: 40.7K (used as training), val: 5K (validation)
            image_size: Target image size (will be resized to this)
            student_processor: TinySiglipProcessor for TinySigLIP model (handles both image and text)
            student_tokenizer: Deprecated - use student_processor instead
            teacher_processor: SigLIPProcessor for teacher model (handles both image and text)
            max_seq_len: Maximum sequence length for text
            cache_dir: Directory to cache the dataset
            use_augmentation: Whether to use data augmentation (only for training)
        """
        if load_dataset is None:
            raise ImportError("datasets library is required. Install it with: pip install datasets")

        self.split = split
        self.image_size = image_size
        self.teacher_processor = teacher_processor
        self.max_seq_len = max_seq_len
        self.use_augmentation = use_augmentation

        # Handle student processor - prefer student_processor over student_tokenizer
        if student_processor is not None:
            self.student_processor = student_processor
        elif student_tokenizer is not None:
            # Backward compatibility: create processor from tokenizer
            print("Warning: student_tokenizer is deprecated. Use student_processor instead.")
            from tinysiglip.processor import TinySiglipProcessor

            self.student_processor = TinySiglipProcessor(
                tokenizer=student_tokenizer,
                image_size=image_size,
                max_seq_len=max_seq_len,
                use_augmentation=use_augmentation,
            )
        else:
            raise ValueError("Either student_processor or student_tokenizer must be provided")

        # Handle split name mapping
        # This dataset only has 'val' and 'test' splits
        # 'test' is the larger split (40.7K), can be used as training data
        # 'val' is smaller (5K), typically used for validation
        if split == "train":
            print("Warning: 'train' split not available in this dataset. Using 'test' split as training data.")
            split = "test"

        # Load dataset from HuggingFace with streaming
        print(f"Loading COCO Caption 2017 {split} split (streaming=True)...")
        self.dataset = load_dataset(
            "lmms-lab/COCO-Caption2017",
            split=split,
            cache_dir=cache_dir,
            streaming=True,
        )
        self.actual_split = split
        print("Using streaming mode - dataset will be loaded on-the-fly")
        print("Note: Each image may have multiple captions, so actual samples will be more")

    def _get_captions(self, item):
        """Extract all captions from a dataset item."""
        captions = []

        # Try different fields that might contain captions
        if "captions" in item:
            captions_raw = item["captions"]
            if isinstance(captions_raw, list):
                captions = captions_raw
            else:
                captions = [captions_raw]
        elif "caption" in item:
            caption = item["caption"]
            if isinstance(caption, list):
                captions = caption
            else:
                captions = [caption]
        elif "text" in item:
            text = item["text"]
            if isinstance(text, list):
                captions = text
            else:
                captions = [text]
        else:
            # Try to find any text field
            text_fields = [k for k in item.keys() if "caption" in k.lower() or "text" in k.lower()]
            if text_fields:
                field_value = item[text_fields[0]]
                if isinstance(field_value, list):
                    captions = field_value
                else:
                    captions = [field_value]

        # Clean and validate captions
        cleaned_captions = []
        for caption in captions:
            if not isinstance(caption, str):
                caption = str(caption)
            caption = caption.strip()
            if len(caption) > 0:
                cleaned_captions.append(caption)

        # Fallback if no valid captions found
        if len(cleaned_captions) == 0:
            cleaned_captions = ["a photo"]

        return cleaned_captions

    def _load_image(self, item):
        """Load and convert image from dataset item to PIL Image."""
        if "image" in item:
            image = item["image"]
            if not isinstance(image, Image.Image):
                # Convert to PIL Image if needed
                image = Image.fromarray(image)
            return image.convert("RGB")
        elif "img_path" in item:
            return Image.open(item["img_path"]).convert("RGB")
        else:
            raise ValueError(f"Cannot find image in dataset item. Available keys: {list(item.keys())}")

    def _process_sample(self, image, caption):
        """Process a single image-caption pair."""
        # Ensure caption is a string and clean it
        if not isinstance(caption, str):
            caption = str(caption)
        caption = caption.strip()
        if len(caption) == 0:
            caption = "a photo"  # Fallback for empty captions

        # Preprocess for TinySigLIP model using TinySiglipProcessor
        student_inputs = self.student_processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
        )
        student_image = student_inputs["pixel_values"].squeeze(0)  # (C, H, W) in [0, 1]
        student_text_ids = student_inputs["input_ids"].squeeze(0)  # (seq_len,)

        # Preprocess for teacher model using SigLIPProcessor
        if self.teacher_processor is not None:
            teacher_inputs = self.teacher_processor(
                text=[caption],
                images=[image],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            teacher_image = teacher_inputs["pixel_values"].squeeze(0)
            teacher_text_ids = teacher_inputs["input_ids"].squeeze(0)
        else:
            teacher_image = student_image
            teacher_text_ids = student_text_ids.clone()

        return {
            "image": student_image,
            "teacher_image": teacher_image,
            "student_text_ids": student_text_ids,
            "teacher_text_ids": teacher_text_ids,
            "caption": caption,
        }

    def __iter__(self):
        """
        Iterate over the dataset, expanding multiple captions per image.
        Each (image, caption) pair is yielded as a separate sample.
        """
        for item in self.dataset:
            # Load image once per item
            try:
                image = self._load_image(item)
            except Exception as e:
                print(f"Warning: Failed to load image: {e}. Skipping item.")
                continue

            # Get all captions for this image
            captions = self._get_captions(item)

            # Yield each (image, caption) pair as a separate sample
            for caption in captions:
                try:
                    yield self._process_sample(image, caption)
                except Exception as e:
                    print(f"Warning: Failed to process sample: {e}. Skipping.")
                    continue


class COCOCaptionDataset(Dataset):
    """
    COCO Caption 2017 dataset for SigLIP training (non-streaming mode).

    Uses regular Dataset for non-streaming data loading.
    Preprocesses all data upfront and stores indices.
    """

    def __init__(
        self,
        split: str = "val",
        image_size: int = 224,
        student_processor: TinySiglipProcessor | None = None,
        student_tokenizer=None,  # Deprecated: use student_processor instead
        teacher_processor=None,
        max_seq_len: int = 77,
        cache_dir: str | None = None,
        use_augmentation: bool = False,
    ):
        """
        Args:
            split: Dataset split ('test', 'val', or 'train' which maps to 'test').
                   test: 40.7K (used as training), val: 5K (validation)
            image_size: Target image size (will be resized to this)
            student_processor: TinySiglipProcessor for TinySigLIP model (handles both image and text)
            student_tokenizer: Deprecated - use student_processor instead
            teacher_processor: SigLIPProcessor for teacher model (handles both image and text)
            max_seq_len: Maximum sequence length for text
            cache_dir: Directory to cache the dataset
            use_augmentation: Whether to use data augmentation (only for training)
        """
        if load_dataset is None:
            raise ImportError("datasets library is required. Install it with: pip install datasets")

        self.split = split
        self.image_size = image_size
        self.teacher_processor = teacher_processor
        self.max_seq_len = max_seq_len
        self.use_augmentation = use_augmentation

        # Handle student processor
        if student_processor is not None:
            self.student_processor = student_processor
        elif student_tokenizer is not None:
            print("Warning: student_tokenizer is deprecated. Use student_processor instead.")
            from tinysiglip.processor import TinySiglipProcessor

            self.student_processor = TinySiglipProcessor(
                tokenizer=student_tokenizer,
                image_size=image_size,
                max_seq_len=max_seq_len,
                use_augmentation=use_augmentation,
            )
        else:
            raise ValueError("Either student_processor or student_tokenizer must be provided")

        # Handle split name mapping
        # This dataset only has 'val' and 'test' splits
        # 'test' is the larger split (40.7K), can be used as training data
        # 'val' is smaller (5K), typically used for validation
        if split == "train":
            print("Warning: 'train' split not available in this dataset. Using 'test' split as training data.")
            split = "test"

        # Load dataset from HuggingFace without streaming
        print(f"Loading COCO Caption 2017 {split} split (streaming=False)...")
        self.hf_dataset = load_dataset(
            "lmms-lab/COCO-Caption2017",
            split=split,
            cache_dir=cache_dir,
            streaming=False,
        )
        self.actual_split = split

        # Generate cache file path based on configuration
        cache_file = self._get_cache_file_path(cache_dir, split, image_size, max_seq_len, use_augmentation)

        # Try to load from cache first
        if cache_file and os.path.exists(cache_file):
            print(f"Loading preprocessed dataset from cache: {cache_file}")
            try:
                with open(cache_file, "rb") as f:
                    self.samples = pickle.load(f)
                print(f"✓ Loaded {len(self.samples)} preprocessed samples from cache")
                return
            except Exception as e:
                print(f"Warning: Failed to load cache file: {e}. Re-preprocessing...")

        # Preprocess all data and create index
        print("Preprocessing dataset (this may take a while)...")
        self.samples = []

        # Get dataset length (non-streaming datasets support len())
        try:
            dataset_len = len(self.hf_dataset)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            dataset_len = None

        for idx, item in enumerate(self.hf_dataset):
            if idx % 1000 == 0 and idx > 0:
                if dataset_len is not None:
                    print(f"  Processed {idx}/{dataset_len} images...")
                else:
                    print(f"  Processed {idx} images...")

            try:
                image = self._load_image(item)
                captions = self._get_captions(item)

                # Store each (image, caption) pair as a separate sample
                for caption in captions:
                    try:
                        sample = self._process_sample(image, caption)
                        self.samples.append(sample)
                    except Exception as e:
                        print(f"Warning: Failed to process sample at idx {idx}: {e}. Skipping.")
                        continue
            except Exception as e:
                print(f"Warning: Failed to load image at idx {idx}: {e}. Skipping.")
                continue

        print(f"✓ Preprocessed {len(self.samples)} samples from {dataset_len if dataset_len else 'unknown'} images")

        # Save to cache for next time
        if cache_file:
            try:
                # Create cache directory if it doesn't exist
                cache_dir_path = os.path.dirname(cache_file)
                if cache_dir_path:
                    os.makedirs(cache_dir_path, exist_ok=True)

                print(f"Saving preprocessed dataset to cache: {cache_file}")
                with open(cache_file, "wb") as f:
                    pickle.dump(self.samples, f)
                print(f"✓ Cached {len(self.samples)} preprocessed samples")
            except Exception as e:
                print(f"Warning: Failed to save cache file: {e}. Continuing without cache...")

    def _get_cache_file_path(self, cache_dir, split, image_size, max_seq_len, use_augmentation):
        """Generate a unique cache file path based on configuration."""
        if cache_dir is None:
            return None

        # Create a hash of the configuration to ensure cache invalidation on config changes
        config_str = f"{split}_{image_size}_{max_seq_len}_{use_augmentation}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        cache_filename = f"coco_preprocessed_{split}_{config_hash}.pkl"
        cache_path = Path(cache_dir) / "preprocessed" / cache_filename

        return str(cache_path)

    def _get_captions(self, item):
        """Extract all captions from a dataset item."""
        captions = []

        # Try different fields that might contain captions
        if "captions" in item:
            captions_raw = item["captions"]
            if isinstance(captions_raw, list):
                captions = captions_raw
            else:
                captions = [captions_raw]
        elif "caption" in item:
            caption = item["caption"]
            if isinstance(caption, list):
                captions = caption
            else:
                captions = [caption]
        elif "text" in item:
            text = item["text"]
            if isinstance(text, list):
                captions = text
            else:
                captions = [text]
        else:
            # Try to find any text field
            text_fields = [k for k in item.keys() if "caption" in k.lower() or "text" in k.lower()]
            if text_fields:
                field_value = item[text_fields[0]]
                if isinstance(field_value, list):
                    captions = field_value
                else:
                    captions = [field_value]

        # Clean and validate captions
        cleaned_captions = []
        for caption in captions:
            if not isinstance(caption, str):
                caption = str(caption)
            caption = caption.strip()
            if len(caption) > 0:
                cleaned_captions.append(caption)

        # Fallback if no valid captions found
        if len(cleaned_captions) == 0:
            cleaned_captions = ["a photo"]

        return cleaned_captions

    def _load_image(self, item):
        """Load and convert image from dataset item to PIL Image."""
        if "image" in item:
            image = item["image"]
            if not isinstance(image, Image.Image):
                # Convert to PIL Image if needed
                image = Image.fromarray(image)
            return image.convert("RGB")
        elif "img_path" in item:
            return Image.open(item["img_path"]).convert("RGB")
        else:
            raise ValueError(f"Cannot find image in dataset item. Available keys: {list(item.keys())}")

    def _process_sample(self, image, caption):
        """Process a single image-caption pair."""
        # Ensure caption is a string and clean it
        if not isinstance(caption, str):
            caption = str(caption)
        caption = caption.strip()
        if len(caption) == 0:
            caption = "a photo"  # Fallback for empty captions

        # Preprocess for TinySigLIP model using TinySiglipProcessor
        student_inputs = self.student_processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
        )
        student_image = student_inputs["pixel_values"].squeeze(0)  # (C, H, W) in [0, 1]
        student_text_ids = student_inputs["input_ids"].squeeze(0)  # (seq_len,)

        # Preprocess for teacher model using SigLIPProcessor
        if self.teacher_processor is not None:
            teacher_inputs = self.teacher_processor(
                text=[caption],
                images=[image],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            teacher_image = teacher_inputs["pixel_values"].squeeze(0)
            teacher_text_ids = teacher_inputs["input_ids"].squeeze(0)
        else:
            teacher_image = student_image
            teacher_text_ids = student_text_ids.clone()

        return {
            "image": student_image,
            "teacher_image": teacher_image,
            "student_text_ids": student_text_ids,
            "teacher_text_ids": teacher_text_ids,
            "caption": caption,
        }

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample by index."""
        return self.samples[idx]


def COCOCaptionDatasetFactory(
    split: str = "val",
    image_size: int = 224,
    student_processor: TinySiglipProcessor | None = None,
    student_tokenizer=None,
    teacher_processor=None,
    max_seq_len: int = 77,
    cache_dir: str | None = None,
    use_augmentation: bool = False,
    streaming: bool = True,
):
    """
    Factory function to create the appropriate dataset class based on streaming mode.

    Args:
        streaming: If True, returns COCOCaptionIterableDataset (streaming mode)
                  If False, returns COCOCaptionDataset (non-streaming mode with preprocessing)
        Other args: Same as dataset classes
    """
    if streaming:
        return COCOCaptionIterableDataset(
            split=split,
            image_size=image_size,
            student_processor=student_processor,
            student_tokenizer=student_tokenizer,
            teacher_processor=teacher_processor,
            max_seq_len=max_seq_len,
            cache_dir=cache_dir,
            use_augmentation=use_augmentation,
        )
    else:
        return COCOCaptionDataset(
            split=split,
            image_size=image_size,
            student_processor=student_processor,
            student_tokenizer=student_tokenizer,
            teacher_processor=teacher_processor,
            max_seq_len=max_seq_len,
            cache_dir=cache_dir,
            use_augmentation=use_augmentation,
        )


# For backward compatibility: COCOCaptionDataset is now a factory function
# It will return the appropriate class based on streaming parameter


def collate_coco_batch(batch):
    """
    Custom collate function for COCO dataset batches.

    Args:
        batch: List of items from COCOCaptionDataset

    Returns:
        dict with batched tensors
    """
    student_images = torch.stack([item["image"] for item in batch])
    teacher_images = torch.stack([item["teacher_image"] for item in batch])
    student_text_ids = torch.stack([item["student_text_ids"] for item in batch])
    teacher_text_ids = torch.stack([item["teacher_text_ids"] for item in batch])
    captions = [item["caption"] for item in batch]

    return {
        "student_images": student_images,
        "teacher_images": teacher_images,
        "student_text_ids": student_text_ids,
        "teacher_text_ids": teacher_text_ids,
        "captions": captions,
    }
