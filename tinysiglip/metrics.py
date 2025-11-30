"""
Evaluation metrics for SigLIP distillation training.
"""

import torch
import torch.nn.functional as F


def compute_recall_at_k(logits_per_image, k_values=(1, 5, 10)):
    """
    Compute Recall@K metrics for image-to-text retrieval.

    Args:
        logits_per_image: (B, B) similarity matrix, where logits_per_image[i, j]
                         is the similarity between image i and text j
        k_values: tuple of K values to compute recall for

    Returns:
        dict: {recall@1, recall@5, recall@10, ...}
    """
    batch_size = logits_per_image.size(0)

    # Ground truth: diagonal is the correct match (image i matches text i)
    # Sort each row to find top-k matches
    _, indices = torch.topk(logits_per_image, k=max(k_values), dim=1)

    # Check if correct match is in top-k
    results = {}
    for k in k_values:
        top_k_indices = indices[:, :k]  # (B, k)
        # Correct match for image i is text i
        correct_indices = torch.arange(batch_size, device=logits_per_image.device).unsqueeze(1)  # (B, 1)
        # Check if correct index is in top-k
        matches = (top_k_indices == correct_indices).any(dim=1)  # (B,)
        recall_at_k = matches.float().mean().item()
        results[f"recall@{k}"] = recall_at_k

    return results


def compute_retrieval_metrics(
    image_features, text_features, logit_scale, prefix=""
):
    """
    Compute retrieval metrics for both image-to-text and text-to-image.

    Args:
        image_features: (B, D) normalized image features
        text_features: (B, D) normalized text features
        logit_scale: scalar temperature scale
        prefix: prefix for metric names (e.g., "student_", "teacher_")

    Returns:
        dict: metrics dictionary
    """
    # Normalize features (safety check)
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # Compute similarity matrices
    logits_per_image = logit_scale * image_features @ text_features.T  # (B, B)
    logits_per_text = logits_per_image.T  # (B, B)

    # Image-to-text retrieval
    i2t_metrics = compute_recall_at_k(logits_per_image, k_values=(1, 5, 10))

    # Text-to-image retrieval
    t2i_metrics = compute_recall_at_k(logits_per_text, k_values=(1, 5, 10))

    # Combine metrics with prefix
    metrics = {}
    for k, v in i2t_metrics.items():
        metrics[f"{prefix}i2t_{k}"] = v
    for k, v in t2i_metrics.items():
        metrics[f"{prefix}t2i_{k}"] = v

    return metrics


def compute_feature_similarity(student_features, teacher_features, prefix=""):
    """
    Compute cosine similarity between student and teacher features.

    Args:
        student_features: (B, D_s) student features
        teacher_features: (B, D_t) teacher features
        prefix: prefix for metric names

    Returns:
        dict: similarity metrics
    """
    # Normalize features
    student_features = F.normalize(student_features, dim=-1)
    teacher_features = F.normalize(teacher_features, dim=-1)

    # Handle dimension mismatch
    student_dim = student_features.shape[-1]
    teacher_dim = teacher_features.shape[-1]

    if student_dim == teacher_dim:
        # Same dimension: direct cosine similarity
        similarities = (student_features * teacher_features).sum(dim=-1)  # (B,)
    elif student_dim < teacher_dim:
        # Student has smaller dimension: compare with first D_s dimensions of teacher
        teacher_features_subset = teacher_features[:, :student_dim]
        similarities = (student_features * teacher_features_subset).sum(dim=-1)  # (B,)
    else:
        # Teacher has smaller dimension: compare with first D_t dimensions of student
        student_features_subset = student_features[:, :teacher_dim]
        similarities = (student_features_subset * teacher_features).sum(dim=-1)  # (B,)

    metrics = {
        f"{prefix}mean_sim": similarities.mean().item(),
        f"{prefix}min_sim": similarities.min().item(),
        f"{prefix}max_sim": similarities.max().item(),
    }

    return metrics


def compute_evaluation_metrics(
    student_image_features,
    student_text_features,
    teacher_image_features,
    teacher_text_features,
    logit_scale,
    projection_v=None,
    projection_t=None,
    student_vision_raw=None,
    student_text_raw=None,
    teacher_vision_raw=None,
    teacher_text_raw=None,
):
    """
    Compute comprehensive evaluation metrics during training.

    Args:
        student_image_features: (B, D_s) student image features (projected)
        student_text_features: (B, D_s) student text features (projected)
        teacher_image_features: (B, D_t) teacher image features (projected)
        teacher_text_features: (B, D_t) teacher text features (projected)
        logit_scale: scalar temperature scale
        projection_v: Optional projection layer for vision features
        projection_t: Optional projection layer for text features
        student_vision_raw: Optional raw student vision features (before projection)
        student_text_raw: Optional raw student text features (before projection)
        teacher_vision_raw: Optional raw teacher vision features (before projection)
        teacher_text_raw: Optional raw teacher text features (before projection)

    Returns:
        dict: comprehensive metrics dictionary
    """
    metrics = {}

    # 1. Student retrieval metrics
    student_metrics = compute_retrieval_metrics(
        student_image_features, student_text_features, logit_scale, prefix="student_"
    )
    metrics.update(student_metrics)

    # 2. Teacher retrieval metrics (for reference)
    teacher_metrics = compute_retrieval_metrics(
        teacher_image_features, teacher_text_features, logit_scale, prefix="teacher_"
    )
    metrics.update(teacher_metrics)

    # 3. Feature similarity (student vs teacher)
    # Use raw features with projections if available for more accurate comparison
    if (
        projection_v is not None
        and student_vision_raw is not None
        and teacher_vision_raw is not None
    ):
        # Project student raw features to teacher dimension
        student_vision_proj = projection_v(student_vision_raw)
        student_vision_proj_norm = F.normalize(student_vision_proj, dim=-1)
        teacher_vision_raw_norm = F.normalize(teacher_vision_raw, dim=-1)
        image_sim = compute_feature_similarity(
            student_vision_proj_norm, teacher_vision_raw_norm, prefix="image_sim_"
        )
    else:
        # Fallback to projected features comparison
        image_sim = compute_feature_similarity(
            student_image_features, teacher_image_features, prefix="image_sim_"
        )
    metrics.update(image_sim)

    if (
        projection_t is not None
        and student_text_raw is not None
        and teacher_text_raw is not None
    ):
        # Project student raw features to teacher dimension
        student_text_proj = projection_t(student_text_raw)
        student_text_proj_norm = F.normalize(student_text_proj, dim=-1)
        teacher_text_raw_norm = F.normalize(teacher_text_raw, dim=-1)
        text_sim = compute_feature_similarity(
            student_text_proj_norm, teacher_text_raw_norm, prefix="text_sim_"
        )
    else:
        # Fallback to projected features comparison
        text_sim = compute_feature_similarity(
            student_text_features, teacher_text_features, prefix="text_sim_"
        )
    metrics.update(text_sim)

    # 4. Cross-modal alignment quality
    # How well does student match teacher in cross-modal retrieval?
    student_logits_per_image = logit_scale * student_image_features @ student_text_features.T
    teacher_logits_per_image = logit_scale * teacher_image_features @ teacher_text_features.T

    # Compute correlation between student and teacher similarity matrices
    student_logits_flat = student_logits_per_image.flatten()
    teacher_logits_flat = teacher_logits_per_image.flatten()

    # Pearson correlation
    student_centered = student_logits_flat - student_logits_flat.mean()
    teacher_centered = teacher_logits_flat - teacher_logits_flat.mean()
    correlation = (student_centered * teacher_centered).sum() / (
        torch.sqrt((student_centered**2).sum() * (teacher_centered**2).sum()) + 1e-8
    )
    metrics["alignment_correlation"] = correlation.item()

    return metrics
