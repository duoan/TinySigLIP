"""
COCO Caption 2017 dataset loader for SigLIP training.
Uses torchvision.datasets.CocoCaptions as the underlying dataset.
Expands multiple captions per image.
"""

import os

import torch
import torchvision.datasets as dset
from PIL import Image
from torch.utils.data import Dataset, IterableDataset

from tinysiglip.processor import TinySiglipProcessor


class COCOCaptionIterableDataset(IterableDataset):
    """
    COCO Caption 2017 dataset for SigLIP training (streaming mode).

    Uses torchvision.datasets.CocoCaptions wrapped as IterableDataset.
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
        coco_root: str | None = None,
        coco_ann_file: str | None = None,
    ):
        """
        Args:
            split: Dataset split ('train', 'val', or 'test').
            image_size: Target image size (will be resized to this)
            student_processor: TinySiglipProcessor for TinySigLIP model (handles both image and text)
            student_tokenizer: Deprecated - use student_processor instead
            teacher_processor: SigLIPProcessor for teacher model (handles both image and text)
            max_seq_len: Maximum sequence length for text
            cache_dir: Directory to cache the dataset (not used with torchvision)
            use_augmentation: Whether to use data augmentation (only for training)
            coco_root: Root directory where COCO images are stored
            coco_ann_file: Path to COCO annotation JSON file
        """
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

        # Validate COCO paths
        if coco_root is None or coco_ann_file is None:
            raise ValueError(
                "coco_root and coco_ann_file must be provided. "
                "Example: coco_root='path/to/coco/images', coco_ann_file='path/to/annotations/captions_train2017.json'"
            )

        if not os.path.exists(coco_root):
            raise ValueError(f"COCO root directory does not exist: {coco_root}")
        if not os.path.exists(coco_ann_file):
            raise ValueError(f"COCO annotation file does not exist: {coco_ann_file}")

        # Load dataset using torchvision
        print(f"Loading COCO Caption 2017 {split} split using torchvision...")
        self.dataset = dset.CocoCaptions(
            root=coco_root,
            annFile=coco_ann_file,
            transform=None,  # We'll handle transforms in _process_sample
        )
        print(f"✓ Loaded {len(self.dataset)} images from COCO dataset")
        print("Note: Each image may have multiple captions, so actual samples will be more")

    def _get_captions(self, target):
        """Extract all captions from torchvision CocoCaptions target (list of strings)."""
        if isinstance(target, list):
            captions = target
        else:
            captions = [target]

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

    def _load_image(self, image):
        """Load and convert image from torchvision CocoCaptions to PIL Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            # Convert tensor or array to PIL Image
            if isinstance(image, torch.Tensor):
                # Convert tensor to PIL
                from torchvision.transforms import ToPILImage

                to_pil = ToPILImage()
                return to_pil(image).convert("RGB")
            else:
                # Assume it's a numpy array
                return Image.fromarray(image).convert("RGB")

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
        """Return the approximate number of samples (may vary due to multiple captions per image)."""
        # Count total captions across all images
        total_samples = 0
        for idx in range(len(self.dataset)):
            try:
                _, target = self.dataset[idx]
                captions = self._get_captions(target)
                total_samples += len(captions)
            except Exception:
                continue
        return total_samples

    def __iter__(self):
        """
        Iterate over the dataset, expanding multiple captions per image.
        Each (image, caption) pair is yielded as a separate sample.
        """
        for idx in range(len(self.dataset)):
            # torchvision CocoCaptions returns (image, target) tuple
            try:
                image, target = self.dataset[idx]
                # Load image
                image = self._load_image(image)
                # Get all captions for this image
                captions = self._get_captions(target)

                # Yield each (image, caption) pair as a separate sample
                for caption in captions:
                    try:
                        yield self._process_sample(image, caption)
                    except Exception as e:
                        print(f"Warning: Failed to process sample at idx {idx}: {e}. Skipping.")
                        continue
            except Exception as e:
                print(f"Warning: Failed to load item at idx {idx}: {e}. Skipping.")
                continue


class COCOCaptionDataset(Dataset):
    """
    COCO Caption 2017 dataset for SigLIP training (non-streaming mode).

    Uses torchvision.datasets.CocoCaptions for data loading.
    Expands multiple captions per image into separate samples.
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
        coco_root: str | None = None,
        coco_ann_file: str | None = None,
    ):
        """
        Args:
            split: Dataset split ('train', 'val', or 'test').
            image_size: Target image size (will be resized to this)
            student_processor: TinySiglipProcessor for TinySigLIP model (handles both image and text)
            student_tokenizer: Deprecated - use student_processor instead
            teacher_processor: SigLIPProcessor for teacher model (handles both image and text)
            max_seq_len: Maximum sequence length for text
            cache_dir: Directory to cache the dataset (not used with torchvision)
            use_augmentation: Whether to use data augmentation (only for training)
            coco_root: Root directory where COCO images are stored
            coco_ann_file: Path to COCO annotation JSON file
        """
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

        # Validate COCO paths
        if coco_root is None or coco_ann_file is None:
            raise ValueError(
                "coco_root and coco_ann_file must be provided. "
                "Example: coco_root='path/to/coco/images', coco_ann_file='path/to/annotations/captions_train2017.json'"
            )

        if not os.path.exists(coco_root):
            raise ValueError(f"COCO root directory does not exist: {coco_root}")
        if not os.path.exists(coco_ann_file):
            raise ValueError(f"COCO annotation file does not exist: {coco_ann_file}")

        # Load dataset using torchvision
        print(f"Loading COCO Caption 2017 {split} split using torchvision...")
        self.coco_dataset = dset.CocoCaptions(
            root=coco_root,
            annFile=coco_ann_file,
            transform=None,  # We'll handle transforms in _process_sample
        )
        print(f"✓ Loaded {len(self.coco_dataset)} images from COCO dataset")

        # Build index mapping: (image_idx, caption_idx) -> sample_idx
        # This allows us to expand multiple captions per image
        self.sample_indices = []
        for img_idx in range(len(self.coco_dataset)):
            image, target = self.coco_dataset[img_idx]
            num_captions = len(target) if isinstance(target, list) else 1
            for cap_idx in range(num_captions):
                self.sample_indices.append((img_idx, cap_idx))

        print(f"✓ Expanded to {len(self.sample_indices)} samples (multiple captions per image)")

    def _get_captions(self, target):
        """Extract all captions from torchvision CocoCaptions target (list of strings)."""
        if isinstance(target, list):
            captions = target
        else:
            captions = [target]

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

    def _load_image(self, image):
        """Load and convert image from torchvision CocoCaptions to PIL Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            # Convert tensor or array to PIL Image
            if isinstance(image, torch.Tensor):
                # Convert tensor to PIL
                from torchvision.transforms import ToPILImage

                to_pil = ToPILImage()
                return to_pil(image).convert("RGB")
            else:
                # Assume it's a numpy array
                return Image.fromarray(image).convert("RGB")

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
        return len(self.sample_indices)

    def __getitem__(self, idx):
        """Get a sample by index."""
        # Get the (image_idx, caption_idx) mapping
        img_idx, cap_idx = self.sample_indices[idx]

        # Load image and captions from COCO dataset
        image, target = self.coco_dataset[img_idx]
        image = self._load_image(image)
        captions = self._get_captions(target)

        # Get the specific caption for this sample
        if cap_idx < len(captions):
            caption = captions[cap_idx]
        else:
            # Fallback to first caption if index is out of range
            caption = captions[0] if captions else "a photo"

        # Process the sample
        return self._process_sample(image, caption)


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
    coco_root: str | None = None,
    coco_ann_file: str | None = None,
):
    """
    Factory function to create the appropriate dataset class based on streaming mode.

    Args:
        streaming: If True, returns COCOCaptionIterableDataset (streaming mode)
                  If False, returns COCOCaptionDataset (non-streaming mode)
        coco_root: Root directory where COCO images are stored
        coco_ann_file: Path to COCO annotation JSON file
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
            coco_root=coco_root,
            coco_ann_file=coco_ann_file,
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
            coco_root=coco_root,
            coco_ann_file=coco_ann_file,
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
