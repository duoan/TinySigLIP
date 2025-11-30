"""
TinySigLIP processor for TinySigLIP model.

Combines image preprocessing and text tokenization for the TinySigLIP model.
This ensures consistent preprocessing between training and inference.
"""

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoTokenizer, PreTrainedTokenizer


class TinySiglipImageProcessor:
    """
    Image processor for TinySigLIP model.

    Handles image preprocessing compatible with timm vision models.
    Uses torchvision transforms internally but can be saved/loaded like HuggingFace processors.
    """

    def __init__(
        self,
        image_size: int = 224,
        use_augmentation: bool = False,
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ):
        """
        Args:
            image_size: Target image size (height, width)
            use_augmentation: Whether to apply data augmentation (for training)
            mean: Normalization mean (if None, no normalization)
            std: Normalization std (if None, no normalization)
        """
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.mean = mean or [0.5, 0.5, 0.5]  # Default to [0.5, 0.5, 0.5] for [0, 1] range
        self.std = std or [0.5, 0.5, 0.5]

        # Create transform pipeline
        self._build_transform()

    def _build_transform(self):
        """Build the image transformation pipeline."""
        transform_list = []

        if self.use_augmentation:
            # Training: add data augmentation
            transform_list.extend(
                [
                    transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                ]
            )
        else:
            # Validation/Test: simple resize
            transform_list.append(transforms.Resize((self.image_size, self.image_size)))

        # Convert to tensor and normalize to [0, 1] range
        transform_list.append(transforms.ToTensor())

        # Optional: normalize (default normalizes [0, 1] to [-1, 1])
        if self.mean and self.std:
            transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))

        self.transform = transforms.Compose(transform_list)

    def __call__(
        self,
        images: Image.Image | list[Image.Image],
        return_tensors: str | None = "pt",
    ) -> dict[str, Any]:
        """
        Process images.

        Args:
            images: Single PIL Image or list of PIL Images
            return_tensors: Return format ("pt" for PyTorch tensors)

        Returns:
            Dict with "pixel_values" key containing processed images
        """
        if isinstance(images, Image.Image):
            images = [images]

        # Convert to RGB if needed
        rgb_images: list[Image.Image] = [img.convert("RGB") for img in images]

        # Apply transforms to each image and collect tensors
        transformed_images: list[torch.Tensor] = []
        for img in rgb_images:
            tensor: torch.Tensor = self.transform(img)  # type: ignore[assignment]
            transformed_images.append(tensor)

        # Stack tensors
        pixel_values = torch.stack(transformed_images)

        if return_tensors == "pt":
            return {"pixel_values": pixel_values}
        else:
            return {"pixel_values": pixel_values.detach().cpu().numpy()}

    def save_pretrained(self, save_directory: str | Path):
        """Save the processor configuration."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        config = {
            "image_size": self.image_size,
            "use_augmentation": self.use_augmentation,
            "mean": self.mean,
            "std": self.std,
            "processor_type": "TinySiglipImageProcessor",
        }

        with open(save_directory / "preprocessor_config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory: str | Path):
        """Load the processor from saved configuration."""
        save_directory = Path(save_directory)
        config_path = save_directory / "preprocessor_config.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}. Use save_pretrained() to save the processor first."
            )

        with open(config_path) as f:
            config = json.load(f)

        return cls(
            image_size=config.get("image_size", 224),
            use_augmentation=config.get("use_augmentation", False),
            mean=config.get("mean"),
            std=config.get("std"),
        )


class TinySiglipProcessor:
    """
    Combined processor for TinySigLIP model.

    Handles both image preprocessing and text tokenization.
    Compatible with HuggingFace save/load format.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        image_processor: TinySiglipImageProcessor | None = None,
        image_size: int = 224,
        max_seq_len: int = 64,
        use_augmentation: bool = False,
    ):
        """
        Args:
            tokenizer: Text tokenizer for TinySigLIP model
            image_processor: Image processor (if None, creates TinySiglipImageProcessor)
            image_size: Target image size
            max_seq_len: Maximum sequence length for text
            use_augmentation: Whether to use data augmentation for images
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if image_processor is None:
            self.image_processor = TinySiglipImageProcessor(image_size=image_size, use_augmentation=use_augmentation)
        else:
            self.image_processor = image_processor

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Image.Image | list[Image.Image] | None = None,
        return_tensors: str | None = "pt",
        padding: bool | str = True,
        truncation: bool = True,
        max_length: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Process text and/or images.

        Args:
            text: Text string(s) to tokenize
            images: Image(s) to process
            return_tensors: Return format ("pt" for PyTorch tensors)
            padding: Padding strategy
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length (uses self.max_seq_len if None)
            **kwargs: Additional arguments for tokenizer

        Returns:
            Dict with processed inputs
        """
        result = {}

        if text is not None:
            # Tokenize text
            tokenizer_kwargs = {
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length or self.max_seq_len,
                "return_tensors": return_tensors,
                **kwargs,
            }
            text_inputs = self.tokenizer(text, **tokenizer_kwargs)
            result.update(text_inputs)

        if images is not None:
            # Process images
            image_inputs = self.image_processor(images, return_tensors=return_tensors)
            result.update(image_inputs)

        return result

    def save_pretrained(self, save_directory: str | Path):
        """Save the processor (tokenizer and image processor) to directory."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save tokenizer
        tokenizer_path = save_directory / "tokenizer"
        self.tokenizer.save_pretrained(str(tokenizer_path))

        # Save image processor
        image_processor_path = save_directory / "image_processor"
        if isinstance(self.image_processor, TinySiglipImageProcessor):
            self.image_processor.save_pretrained(image_processor_path)
        else:
            # HuggingFace ImageProcessor
            self.image_processor.save_pretrained(str(image_processor_path))

        # Save processor config
        config = {
            "processor_type": "TinySiglipProcessor",
            "max_seq_len": self.max_seq_len,
            "tokenizer_class": type(self.tokenizer).__name__,
            "image_processor_class": type(self.image_processor).__name__,
        }

        with open(save_directory / "processor_config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        save_directory: str | Path,
        use_augmentation: bool = False,
    ):
        """
        Load the processor from saved directory.

        Args:
            save_directory: Directory containing saved processor
            use_augmentation: Whether to use augmentation (overrides saved config for training)
        """
        save_directory = Path(save_directory)

        # Load tokenizer
        tokenizer_path = save_directory / "tokenizer"
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. Use save_pretrained() to save the processor first."
            )
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        # Load image processor
        image_processor_path = save_directory / "image_processor"
        if not image_processor_path.exists():
            raise FileNotFoundError(
                f"Image processor not found at {image_processor_path}. "
                "Use save_pretrained() to save the processor first."
            )

        # Check if it's our custom TinySiglipImageProcessor or HuggingFace ImageProcessor
        config_path = image_processor_path / "preprocessor_config.json"
        if config_path.exists():
            with open(config_path) as f:
                img_config = json.load(f)
            if img_config.get("processor_type") == "TinySiglipImageProcessor":
                image_processor = TinySiglipImageProcessor.from_pretrained(image_processor_path)
                # Override augmentation if specified
                if use_augmentation != image_processor.use_augmentation:
                    image_processor.use_augmentation = use_augmentation
                    image_processor._build_transform()
            else:
                # HuggingFace ImageProcessor
                image_processor = AutoImageProcessor.from_pretrained(str(image_processor_path))
        else:
            # Try loading as HuggingFace ImageProcessor
            image_processor = AutoImageProcessor.from_pretrained(str(image_processor_path))

        # Load processor config
        config_path = save_directory / "processor_config.json"
        max_seq_len = 64  # default
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                max_seq_len = config.get("max_seq_len", 64)

        return cls(
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_seq_len=max_seq_len,
        )
