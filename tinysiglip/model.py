import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import timm
import torch
import torch.nn as nn


@dataclass
class TinySiglipConfig:
    """
    Configuration for TinySiglipModel.
    Designed to distill SigLIP 2 into a compact student architecture.
    """

    # Vision model configuration
    # Defaulting to ViT-Tiny/Small as per the student architecture plan.
    vision_model_name: str = "vit_tiny_patch16_224"

    # Text model configuration
    # Using a smaller transformer for the student (e.g., 4 layers).
    # Vocab size should match the teacher's (SigLIP 2 uses Gemma tokenizer -> 256k).
    # If using a subset or different tokenizer, adjust accordingly.
    text_vocab_size: int = 32000
    text_seq_len: int = 64
    text_dim: int = 384
    text_layers: int = 4
    text_nhead: int = 8
    text_ff_dim_multiplier: int = 4

    # Projection configuration
    # Dimensions to project both vision and text features into.
    projection_dim: int = 384

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TinySiglipConfig":
        return cls(**config_dict)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "TinySiglipConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class VisionModel(nn.Module):
    """
    Vision encoder wrapper.
    Uses `timm` to instantiate the backbone and adds a projection layer.
    """

    def __init__(
        self,
        vision_model_name: str = "vit_tiny_patch16_224",
        projection_dim: int = 384,
    ):
        super().__init__()

        # Initialize vision backbone from timm
        # num_classes=0 ensures we get the raw feature embeddings (before classifier)
        self.encoder = timm.create_model(vision_model_name, pretrained=True, num_classes=0)
        vision_feat_dim: int = cast(int, self.encoder.num_features)

        # Projection layer to align vision features with text features
        self.projection = nn.Linear(vision_feat_dim, projection_dim, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) normalized image tensor.
        Returns:
            image_features: (B, projection_dim) projected features.
        """
        # Extract features from backbone
        vision_features = self.encoder(images)

        # Handle different output shapes from various backbones
        if vision_features.dim() == 4:  # CNN-like (B, C, H, W)
            vision_raw = vision_features.mean(dim=[2, 3])  # Global Average Pooling
        elif vision_features.dim() == 3:  # Transformer-like (B, Seq, D)
            vision_raw = vision_features[:, 0, :]  # Take CLS token
        else:  # (B, D) already pooled
            vision_raw = vision_features

        # Project to shared embedding space
        image_features = self.projection(vision_raw)
        return image_features


class TextModel(nn.Module):
    """
    Custom lightweight Transformer Encoder for text.
    Designed to be small and efficient for the student model.
    """

    def __init__(
        self,
        text_vocab_size: int = 32000,
        text_seq_len: int = 64,
        text_dim: int = 384,
        text_layers: int = 4,
        text_nhead: int = 8,
        text_ff_dim_multiplier: int = 4,
        projection_dim: int = 384,
    ):
        super().__init__()
        self.text_seq_len = text_seq_len

        # Token Embeddings
        self.embedding = nn.Embedding(text_vocab_size, text_dim)

        # Learnable Positional Embeddings
        # Initialized with small std to aid convergence
        self.pos_embedding = nn.Parameter(torch.randn(1, text_seq_len, text_dim) * 0.02)

        # Transformer Encoder
        # batch_first=True for modern PyTorch usage
        # norm_first=True (Pre-LN) is generally more stable for training
        text_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=text_nhead,
            dim_feedforward=text_dim * text_ff_dim_multiplier,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(text_layer, num_layers=text_layers)

        # Projection layer
        self.projection = nn.Linear(text_dim, projection_dim, bias=False)

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for stability."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, text_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            text_ids: (B, Seq_Len) token indices.
            attention_mask: (B, Seq_Len) 1 for valid tokens, 0 for padding.
        Returns:
            text_features: (B, projection_dim) projected features.
        """
        B, L = text_ids.shape

        # 1. Embedding lookup
        x = self.embedding(text_ids)

        # 2. Add Positional Embeddings
        # Safe slicing to handle cases where input length < config length
        if L <= self.pos_embedding.shape[1]:
            x = x + self.pos_embedding[:, :L, :]
        else:
            x = x + self.pos_embedding[:, :L, :]

        # 3. Create Padding Mask for Transformer
        # TransformerEncoder expects True for padded positions to be ignored.
        # Input mask is usually 1=keep, 0=pad. So we invert it.
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        # 4. Transformer Forward Pass
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # 5. Pooling (Take the first token / CLS-equivalent)
        # Assuming the tokenizer places a start token at index 0, or the model learns to compress info there.
        text_raw = x[:, 0, :]

        # 6. Project
        text_features = self.projection(text_raw)
        return text_features


class TinySiglipModel(nn.Module):
    """
    Main Student Model combining Vision and Text towers.
    Includes the specific learnable Scale and Bias parameters required for SigLIP loss.
    """

    def __init__(self, config: TinySiglipConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = TinySiglipConfig(**kwargs)
        self.config = config

        # Instantiate Vision Tower
        self.vision_model = VisionModel(
            vision_model_name=config.vision_model_name,
            projection_dim=config.projection_dim,
        )

        # Instantiate Text Tower
        self.text_model = TextModel(
            text_vocab_size=config.text_vocab_size,
            text_seq_len=config.text_seq_len,
            text_dim=config.text_dim,
            text_layers=config.text_layers,
            text_nhead=config.text_nhead,
            text_ff_dim_multiplier=config.text_ff_dim_multiplier,
            projection_dim=config.projection_dim,
        )

        # --- SigLIP Specific Parameters ---

        # logit_scale: Controls the temperature of the sigmoid loss.
        # Initialized to ln(10) ≈ 2.3026. This is a learnable parameter.
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.3026)

        # logit_bias: A learned bias added to the logits before sigmoid.
        # SigLIP initializes this to -10.0 to handle the heavy class imbalance (mostly negatives).
        self.logit_bias = nn.Parameter(torch.ones([]) * -10.0)

    def forward(
        self, images: torch.Tensor, text_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        During training, returns features + current scale/bias for loss calculation.
        During inference, returns only features.

        Args:
            images: (B, 3, H, W)
            text_ids: (B, Seq)
            attention_mask: (B, Seq)
        """
        image_features = self.vision_model(images)
        text_features = self.text_model(text_ids, attention_mask=attention_mask)

        if self.training:
            # Important: Return exp(scale) to ensure the temperature is positive
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        else:
            return image_features, text_features
