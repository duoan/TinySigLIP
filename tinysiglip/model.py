"""
Simplified SigLIP distillation model using timm.
"""

from typing import cast

import timm
import torch
import torch.nn as nn


class TinySiglipModel(nn.Module):
    """TinySiglip model with vision and text encoders for SigLIP distillation."""

    def __init__(
        self,
        vision_model_name: str = "vit_tiny_patch16_224",
        vision_dim: int = 384,
        text_vocab_size: int = 32000,
        text_seq_len: int = 64,
        text_dim: int = 384,
        text_layers: int = 4,
        text_nhead: int = 8,
        text_ff_dim_multiplier: int = 4,
        projection_dim: int = 384,
    ):
        super().__init__()

        # Vision encoder from timm
        self.vision_backbone = timm.create_model(vision_model_name, pretrained=True, num_classes=0)
        vision_feat_dim: int = cast(int, self.vision_backbone.num_features)

        # Vision projection
        self.vision_proj = nn.Linear(vision_feat_dim, projection_dim, bias=False)

        # Text encoder (simple transformer)
        self.text_embedding = nn.Embedding(text_vocab_size, text_dim)
        self.text_pos_embedding = nn.Parameter(torch.randn(1, text_seq_len, text_dim))

        text_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=text_nhead,
            dim_feedforward=text_dim * text_ff_dim_multiplier,
            batch_first=True,
            norm_first=True,
        )
        self.text_transformer = nn.TransformerEncoder(text_layer, num_layers=text_layers)

        # Text projection
        self.text_proj = nn.Linear(text_dim, projection_dim, bias=False)

    def forward(self, images, text_ids):
        """
        Args:
            images: (B, 3, H, W)
            text_ids: (B, seq_len)

        Returns:
            image_features: (B, projection_dim)
            text_features: (B, projection_dim)
            vision_raw: (B, vision_feat_dim) - raw vision features for distillation
            text_raw: (B, text_dim) - raw text features for distillation
        """
        # Vision encoder
        vision_features = self.vision_backbone(images)

        # Handle different output formats from timm
        if vision_features.dim() == 4:  # CNN output (B, C, H, W)
            vision_raw = vision_features.mean(dim=[2, 3])  # Global average pooling
        elif vision_features.dim() == 3:  # ViT output (B, L, D)
            vision_raw = vision_features[:, 0, :]  # CLS token
        else:  # (B, D)
            vision_raw = vision_features

        image_features = self.vision_proj(vision_raw)

        # Text encoder
        text_embeds = self.text_embedding(text_ids)  # (B, seq_len, text_dim)
        text_embeds = text_embeds + self.text_pos_embedding

        text_features_all = self.text_transformer(text_embeds)  # (B, seq_len, text_dim)
        text_raw = text_features_all[:, 0, :]  # CLS token
        text_features = self.text_proj(text_raw)

        return image_features, text_features, vision_raw, text_raw
