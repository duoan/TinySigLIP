"""
COCO Caption 2017 dataset loader for SigLIP training.
Uses IterableDataset for streaming and expands multiple captions per image.
"""

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import IterableDataset

from tinysiglip.processor import TinySiglipProcessor


class COCOCaptionDataset(IterableDataset):
    """
    COCO Caption 2017 dataset for SigLIP training.

    Loads image-text pairs from COCO Caption dataset and tokenizes them
    using student and teacher tokenizers.
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
        streaming: bool = True,
    ):
        """
        Args:
            split: Dataset split ('test', 'val'), test: 40.7K, val: 5K
            image_size: Target image size (will be resized to this)
            student_processor: TinySiglipProcessor for TinySigLIP model (handles both image and text)
            student_tokenizer: Deprecated - use student_processor instead
            teacher_processor: SigLIPProcessor for teacher model (handles both image and text)
            max_seq_len: Maximum sequence length for text
            cache_dir: Directory to cache the dataset
            use_augmentation: Whether to use data augmentation (only for training)
            streaming: Whether to use streaming mode (default True for memory efficiency)
        """
        if load_dataset is None:
            raise ImportError("datasets library is required. Install it with: pip install datasets")

        self.split = split
        self.image_size = image_size
        self.teacher_processor = teacher_processor
        self.max_seq_len = max_seq_len
        self.use_augmentation = use_augmentation
        self.streaming = streaming

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

        # Handle split name mapping without downloading the dataset
        # This dataset only has 'val' and 'test' splits
        # If user requests 'train', automatically use 'val' instead
        if split == "train":
            print("Warning: 'train' split not available in this dataset. Using 'val' split instead.")
            split = "val"

        # Load dataset from HuggingFace with streaming
        print(f"Loading COCO Caption 2017 {split} split (streaming={streaming})...")
        self.dataset = load_dataset(
            "lmms-lab/COCO-Caption2017",
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
        )
        # Store the actual split used
        self.actual_split = split
        if not streaming:
            try:
                # Try to get length if available (not available for IterableDataset)
                dataset_len = len(self.dataset)  # type: ignore[arg-type]
                print(f"Loaded {dataset_len} image samples")
            except (TypeError, AttributeError):
                print("Dataset loaded (size unknown)")
            print("Note: Each image may have multiple captions, so actual samples will be more")
        else:
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
        # IMPORTANT: Use processor to ensure format matches exactly what SigLIP expects
        if self.teacher_processor is not None:
            # Process with teacher processor (expects PIL Image and text string)
            teacher_inputs = self.teacher_processor(
                text=[caption],
                images=[image],  # PIL Image (processor will handle resize/normalize)
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            # Processor returns pixel_values and input_ids
            teacher_image = teacher_inputs["pixel_values"].squeeze(0)  # (C, H, W) in [0, 1]
            teacher_text_ids = teacher_inputs["input_ids"].squeeze(0)  # (seq_len,)
        else:
            # Fallback: use student preprocessing if no processor
            teacher_image = student_image
            teacher_text_ids = student_text_ids.clone()

        return {
            "image": student_image,  # Student image (from processor)
            "teacher_image": teacher_image,  # Teacher image (from processor)
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
