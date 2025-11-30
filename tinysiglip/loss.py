"""
Loss functions for SigLIP distillation.
"""

import torch
import torch.nn.functional as F


def siglip_loss(logits_per_image, logits_per_text):
    """
    SigLIP loss: binary cross-entropy with sigmoid (not softmax).

    Args:
        logits_per_image: (B, B) similarity matrix
        logits_per_text: (B, B) similarity matrix (transpose of above)

    Returns:
        loss: scalar
    """
    labels = torch.eye(logits_per_image.size(0), device=logits_per_image.device)

    # Image-to-text loss
    loss_i2t = F.binary_cross_entropy_with_logits(logits_per_image, labels, reduction="mean")

    # Text-to-image loss
    loss_t2i = F.binary_cross_entropy_with_logits(logits_per_text, labels, reduction="mean")

    return (loss_i2t + loss_t2i) / 2.0


def cross_modal_distillation_loss(teacher_logits, student_logits, temperature=3.0):
    """
    Cross-Modal Distillation (CMD) loss: KL divergence between teacher and student distributions.

    Args:
        teacher_logits: (B, B) teacher similarity matrix
        student_logits: (B, B) student similarity matrix
        temperature: temperature for softmax

    Returns:
        loss: scalar
    """
    # For distillation, we use softmax to create probability distributions
    # (even though SigLIP uses sigmoid for its own loss)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)

    # KL divergence: KL(teacher || student)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean", log_target=False)

    return kl_loss * (temperature**2)


def feature_distillation_loss(teacher_feat, student_feat):
    """
    Uni-Modal Distillation (UMD) loss: MSE between teacher and student features.

    Args:
        teacher_feat: (B, D) teacher features
        student_feat: (B, D) student features (already projected to teacher dimension)

    Returns:
        loss: scalar
    """
    return F.mse_loss(student_feat, teacher_feat)


def compute_total_loss(
    student_image_features,
    student_text_features,
    teacher_image_features,
    teacher_text_features,
    teacher_vision_raw,
    teacher_text_raw,
    student_vision_raw,
    student_text_raw,
    projection_v,
    projection_t,
    logit_scale,
    lambda_siglip=0.5,
    lambda_cmd=1.0,
    lambda_umd=0.5,
    temperature=3.0,
):
    """
    Compute total distillation loss.

    Note: Embedding loss is not included here because we use weight transfer
    (one-time initialization) instead of ongoing embedding mimicking loss.

    Args:
        student_image_features: (B, D) student image features
        student_text_features: (B, D) student text features
        teacher_image_features: (B, D) teacher image features
        teacher_text_features: (B, D) teacher text features
        teacher_vision_raw: (B, D_t) teacher raw vision features
        teacher_text_raw: (B, D_t) teacher raw text features
        student_vision_raw: (B, D_s) student raw vision features
        student_text_raw: (B, D_s) student raw text features
        projection_v: nn.Linear to project student vision to teacher dimension
        projection_t: nn.Linear to project student text to teacher dimension
        logit_scale: temperature scale
        lambda_cmd: weight for CMD loss
        lambda_umd: weight for UMD loss
        temperature: temperature for distillation

    Returns:
        total_loss: scalar
        loss_dict: dictionary with individual losses
    """
    # Normalize features
    student_image_features = F.normalize(student_image_features, dim=-1)
    student_text_features = F.normalize(student_text_features, dim=-1)
    teacher_image_features = F.normalize(teacher_image_features, dim=-1)
    teacher_text_features = F.normalize(teacher_text_features, dim=-1)

    # Compute similarity matrices
    student_logits_per_image = logit_scale * student_image_features @ student_text_features.T
    student_logits_per_text = student_logits_per_image.T

    teacher_logits_per_image = logit_scale * teacher_image_features @ teacher_text_features.T
    teacher_logits_per_text = teacher_logits_per_image.T

    # 1. SigLIP loss (self-supervised)
    loss_siglip = siglip_loss(student_logits_per_image, student_logits_per_text)

    # 2. Cross-modal distillation loss
    loss_cmd = (
        cross_modal_distillation_loss(teacher_logits_per_image, student_logits_per_image, temperature)
        + cross_modal_distillation_loss(teacher_logits_per_text, student_logits_per_text, temperature)
    ) / 2.0

    # 3. Uni-modal distillation loss
    teacher_vision_raw = F.normalize(teacher_vision_raw, dim=-1)
    teacher_text_raw = F.normalize(teacher_text_raw, dim=-1)
    student_vision_raw = F.normalize(student_vision_raw, dim=-1)
    student_text_raw = F.normalize(student_text_raw, dim=-1)

    student_vision_proj = projection_v(student_vision_raw)
    student_text_proj = projection_t(student_text_raw)

    loss_umd_vision = feature_distillation_loss(teacher_vision_raw, student_vision_proj)
    loss_umd_text = feature_distillation_loss(teacher_text_raw, student_text_proj)
    loss_umd = (loss_umd_vision + loss_umd_text) / 2.0

    # Total loss (no embedding loss - we use weight transfer instead)
    loss_weighted = lambda_siglip * loss_siglip + lambda_cmd * loss_cmd + lambda_umd * loss_umd

    loss_dict = {
        "siglip": loss_siglip.item(),
        "cmd": loss_cmd.item(),
        "umd": loss_umd.item(),
        "weighted": loss_weighted.item(),
    }

    return loss_weighted, loss_dict
