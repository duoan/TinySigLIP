"""
COCO Caption 2017 dataset loader for SigLIP training.
Uses torchvision.datasets.CocoCaptions as the underlying dataset.
Expands multiple captions per image.
"""

import hashlib
import os
import pickle
import time
from pathlib import Path

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
        verbose: bool = True,  # Whether to print verbose messages (set to False in distributed training)
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
            verbose: Whether to print verbose messages (set to False in distributed training)
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
            if verbose:
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

        # Determine the correct image directory based on split
        # COCO images are organized as: images/train2017/, images/val2017/, images/test2017/
        # torchvision CocoCaptions expects root to point to the images directory,
        # and annotation files contain paths like "train2017/xxxx.jpg"
        # So we keep coco_root as the images directory (not the specific split directory)
        if not os.path.exists(coco_root):
            raise ValueError(f"COCO root directory does not exist: {coco_root}")
        if not os.path.exists(coco_ann_file):
            raise ValueError(f"COCO annotation file does not exist: {coco_ann_file}")

        # Verify that the split directory exists
        split_dir = os.path.join(coco_root, f"{split}2017")
        if not os.path.exists(split_dir):
            raise ValueError(
                f"COCO split directory does not exist: {split_dir}\n"
                f"Expected structure: {coco_root}/{{train2017,val2017,test2017}}/"
            )

        # Load dataset using torchvision
        # For COCO, the annotation file contains paths like "train2017/xxxx.jpg"
        # If root points to images/, torchvision will look for images/train2017/xxxx.jpg
        # However, some annotation files might have paths without the split prefix
        # So we try both: root pointing to images/ and root pointing to the split directory
        if verbose:
            print(f"Loading COCO Caption 2017 {split} split using torchvision...")
            print(f"  Image root: {coco_root}")
            print(f"  Annotation file: {coco_ann_file}")
            print(f"  Split directory: {split_dir}")

        # Try with root pointing to images/ directory first (standard COCO format)
        try:
            self.dataset = dset.CocoCaptions(
                root=coco_root,  # Points to images/ directory
                annFile=coco_ann_file,
                transform=None,
            )
            # Test if we can actually load an image
            if len(self.dataset) > 0:
                try:
                    _ = self.dataset[0]
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Failed to load image with root={coco_root}: {e}")
                        print("  Trying with root pointing to split directory...")
                    # Fallback: try with root pointing to split directory
                    self.dataset = dset.CocoCaptions(
                        root=split_dir,  # Points to images/train2017/ directory
                        annFile=coco_ann_file,
                        transform=None,
                    )
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to load with root={coco_root}: {e}")
                print("  Trying with root pointing to split directory...")
            # Fallback: try with root pointing to split directory
            self.dataset = dset.CocoCaptions(
                root=split_dir,  # Points to images/train2017/ directory
                annFile=coco_ann_file,
                transform=None,
            )
        if verbose:
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
        verbose: bool = True,  # Whether to print verbose messages (set to False in distributed training)
        is_main_process: bool = True,  # Whether this is the main process (for distributed training)
    ):
        """
        Args:
            split: Dataset split ('train', 'val', or 'test').
            image_size: Target image size (will be resized to this)
            student_processor: TinySiglipProcessor for TinySigLIP model (handles both image and text)
            student_tokenizer: Deprecated - use student_processor instead
            teacher_processor: SigLIPProcessor for teacher model (handles both image and text)
            max_seq_len: Maximum sequence length for text
            cache_dir: Directory to cache the dataset and index (for distributed training)
            use_augmentation: Whether to use data augmentation (only for training)
            coco_root: Root directory where COCO images are stored
            coco_ann_file: Path to COCO annotation JSON file
            verbose: Whether to print verbose messages (set to False in distributed training)
            is_main_process: Whether this is the main process (for distributed training)
        """
        self.split = split
        self.image_size = image_size
        self.teacher_processor = teacher_processor
        self.max_seq_len = max_seq_len
        self.use_augmentation = use_augmentation
        self.verbose = verbose
        self.cache_dir = cache_dir
        self.is_main_process = is_main_process
        self.coco_root = coco_root  # Store for cache path generation
        self.coco_ann_file = coco_ann_file  # Store for cache path generation
        self.cache_dir = cache_dir
        self.is_main_process = is_main_process
        self.coco_root = coco_root  # Store for cache path generation
        self.coco_ann_file = coco_ann_file  # Store for cache path generation

        # Handle student processor
        if student_processor is not None:
            self.student_processor = student_processor
        elif student_tokenizer is not None:
            if verbose:
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

        # Determine the correct image directory based on split
        # COCO images are organized as: images/train2017/, images/val2017/, images/test2017/
        # torchvision CocoCaptions expects root to point to the images directory,
        # and annotation files contain paths like "train2017/xxxx.jpg"
        # So we keep coco_root as the images directory (not the specific split directory)
        if not os.path.exists(coco_root):
            raise ValueError(f"COCO root directory does not exist: {coco_root}")
        if not os.path.exists(coco_ann_file):
            raise ValueError(f"COCO annotation file does not exist: {coco_ann_file}")

        # Verify that the split directory exists
        split_dir = os.path.join(coco_root, f"{split}2017")
        if not os.path.exists(split_dir):
            raise ValueError(
                f"COCO split directory does not exist: {split_dir}\n"
                f"Expected structure: {coco_root}/{{train2017,val2017,test2017}}/"
            )

        # Load dataset using torchvision
        # For COCO, the annotation file contains paths like "train2017/xxxx.jpg"
        # If root points to images/, torchvision will look for images/train2017/xxxx.jpg
        # However, some annotation files might have paths without the split prefix
        # So we try both: root pointing to images/ and root pointing to the split directory
        if verbose:
            print(f"Loading COCO Caption 2017 {split} split using torchvision...")
            print(f"  Image root: {coco_root}")
            print(f"  Annotation file: {coco_ann_file}")
            print(f"  Split directory: {split_dir}")

        # Try with root pointing to images/ directory first (standard COCO format)
        try:
            self.coco_dataset = dset.CocoCaptions(
                root=coco_root,  # Points to images/ directory
                annFile=coco_ann_file,
                transform=None,
            )
            # Test if we can actually load an image
            if len(self.coco_dataset) > 0:
                try:
                    _ = self.coco_dataset[0]
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Failed to load image with root={coco_root}: {e}")
                        print("  Trying with root pointing to split directory...")
                    # Fallback: try with root pointing to split directory
                    self.coco_dataset = dset.CocoCaptions(
                        root=split_dir,  # Points to images/train2017/ directory
                        annFile=coco_ann_file,
                        transform=None,
                    )
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to load with root={coco_root}: {e}")
                print("  Trying with root pointing to split directory...")
            # Fallback: try with root pointing to split directory
            self.coco_dataset = dset.CocoCaptions(
                root=split_dir,  # Points to images/train2017/ directory
                annFile=coco_ann_file,
                transform=None,
            )
        if verbose:
            print(f"✓ Loaded {len(self.coco_dataset)} images from COCO dataset")

        # Build index mapping: (image_idx, caption_idx) -> sample_idx
        # This allows us to expand multiple captions per image
        # In distributed training, only main process builds the index and caches it
        # Other processes wait and load from cache
        self.sample_indices = self._build_or_load_index(
            cache_dir=self.cache_dir,
            is_main_process=self.is_main_process,
            verbose=verbose,
        )

        if verbose:
            print(f"✓ Expanded to {len(self.sample_indices)} samples (multiple captions per image)")

    def _get_index_cache_path(self, cache_dir: str | None) -> Path | None:
        """Generate cache file path for the index."""
        if cache_dir is None:
            return None

        # Create a hash based on dataset configuration to ensure cache invalidation
        config_str = f"{self.split}_{self.coco_root}_{self.coco_ann_file}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        cache_file = Path(cache_dir) / "coco_index_cache" / f"index_{self.split}_{config_hash}.pkl"
        return cache_file

    def _build_or_load_index(
        self, cache_dir: str | None, is_main_process: bool, verbose: bool
    ) -> list[tuple[int, int]]:
        """
        Build index mapping or load from cache.

        In distributed training:
        - Main process builds the index and saves to cache
        - Other processes wait for cache file and load it
        """
        cache_path = self._get_index_cache_path(cache_dir)

        # Try to load from cache first
        if cache_path and cache_path.exists():
            if verbose:
                print(f"Loading index from cache: {cache_path}")
            try:
                with open(cache_path, "rb") as f:
                    sample_indices = pickle.load(f)
                if verbose:
                    print(f"✓ Loaded {len(sample_indices)} index entries from cache")
                return sample_indices
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to load cache: {e}. Rebuilding index...")

        # Build index
        if is_main_process:
            if verbose:
                print("Building index mapping (this may take a while)...")
            sample_indices = []
            for img_idx in range(len(self.coco_dataset)):
                if verbose and img_idx % 1000 == 0 and img_idx > 0:
                    print(f"  Processing image {img_idx}/{len(self.coco_dataset)}...")
                try:
                    _, target = self.coco_dataset[img_idx]
                    num_captions = len(target) if isinstance(target, list) else 1
                    for cap_idx in range(num_captions):
                        sample_indices.append((img_idx, cap_idx))
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Failed to process image {img_idx}: {e}. Skipping.")
                    continue

            # Save to cache
            if cache_path:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "wb") as f:
                        pickle.dump(sample_indices, f)
                    if verbose:
                        print(f"✓ Saved index to cache: {cache_path}")
                except Exception as e:
                    if verbose:
                        print(f"Warning: Failed to save cache: {e}")

            return sample_indices
        else:
            # Non-main process: wait for cache file to be created
            if cache_path:
                if verbose:
                    print("Waiting for index cache to be created by main process...")
                max_wait_time = 300  # 5 minutes
                wait_interval = 1  # 1 second
                waited = 0
                while waited < max_wait_time:
                    if cache_path.exists():
                        try:
                            with open(cache_path, "rb") as f:
                                sample_indices = pickle.load(f)
                            if verbose:
                                print(f"✓ Loaded {len(sample_indices)} index entries from cache")
                            return sample_indices
                        except Exception:
                            # File might be incomplete, wait a bit more
                            time.sleep(wait_interval)
                            waited += wait_interval
                            continue
                    time.sleep(wait_interval)
                    waited += wait_interval

                # If cache still doesn't exist after waiting, build it ourselves
                if verbose:
                    print("Warning: Cache file not found after waiting. Building index on this process...")
            else:
                # No cache directory, build index anyway
                if verbose:
                    print("No cache directory specified. Building index on this process...")

            # Fallback: build index on this process too
            sample_indices = []
            for img_idx in range(len(self.coco_dataset)):
                try:
                    _, target = self.coco_dataset[img_idx]
                    num_captions = len(target) if isinstance(target, list) else 1
                    for cap_idx in range(num_captions):
                        sample_indices.append((img_idx, cap_idx))
                except Exception as e:
                    if verbose:
                        print(f"Warning: Failed to process image {img_idx}: {e}. Skipping.")
                    continue
            return sample_indices

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
    verbose: bool = True,  # Whether to print verbose messages (set to False in distributed training)
    is_main_process: bool = True,  # Whether this is the main process (for distributed training)
):
    """
    Factory function to create the appropriate dataset class based on streaming mode.

    Args:
        streaming: If True, returns COCOCaptionIterableDataset (streaming mode)
                  If False, returns COCOCaptionDataset (non-streaming mode)
        coco_root: Root directory where COCO images are stored
        coco_ann_file: Path to COCO annotation JSON file
        verbose: Whether to print verbose messages (set to False in distributed training)
        is_main_process: Whether this is the main process (for distributed training)
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
            verbose=verbose,
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
            verbose=verbose,
            is_main_process=is_main_process,
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
