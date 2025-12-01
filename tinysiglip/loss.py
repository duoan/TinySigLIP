import torch
import torch.nn.functional as F


def siglip_loss(logits_per_image, logits_per_text):
    """
    Computes the SigLIP loss using Binary Cross-Entropy (BCE).
    SigLIP treats the image-text matching problem as independent binary classification
    for every pair in the batch, rather than a softmax classification.

    Args:
        logits_per_image: (B, B) tensor, similarity matrix (image x text).
        logits_per_text: (B, B) tensor, similarity matrix (text x image).

    Returns:
        loss: Scalar tensor representing the average SigLIP loss.
    """
    # 1. Create labels: diagonal elements are 1 (positives), off-diagonals are 0 (negatives).
    # tensor shape: (Batch_Size, Batch_Size)
    labels = torch.eye(logits_per_image.size(0), device=logits_per_image.device)

    # 2. Compute BCE loss for Image-to-Text direction
    # F.binary_cross_entropy_with_logits includes the sigmoid activation internally.
    loss_i2t = F.binary_cross_entropy_with_logits(logits_per_image, labels, reduction="mean")

    # 3. Compute BCE loss for Text-to-Image direction
    loss_t2i = F.binary_cross_entropy_with_logits(logits_per_text, labels, reduction="mean")

    # 4. Average the bidirectional losses
    return (loss_i2t + loss_t2i) / 2.0


def cross_modal_distillation_loss(teacher_logits, student_logits, temperature=3.0):
    """
    Computes the Cross-Modal Distillation (CMD) loss using KL Divergence.
    This aligns the student's similarity distribution (soft labels) with the teacher's.

    Args:
        teacher_logits: (B, B) tensor, teacher's similarity scores.
        student_logits: (B, B) tensor, student's similarity scores.
        temperature: Float, scaling factor to soften the distributions.

    Returns:
        loss: Scalar tensor representing the scaled KL divergence loss.
    """
    # 1. Compute Teacher's probability distribution (Target)
    # Use Softmax (not Sigmoid) here to represent the relative ranking/distribution across the batch.
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

    # 2. Compute Student's log-probability distribution
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)

    # 3. Compute KL Divergence: KL(Teacher || Student)
    # reduction="batchmean" mathematically computes the true mean of KL divergence over the batch
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean", log_target=False)

    # 4. Scale gradients by T^2 (standard practice in Knowledge Distillation)
    return kl_loss * (temperature**2)


def feature_distillation_loss(teacher_feat, student_feat):
    """
    Computes the Uni-Modal Distillation (UMD) loss using Mean Squared Error (MSE).
    This forces the student's embeddings to strictly mimic the teacher's embeddings.

    Args:
        teacher_feat: (B, D) tensor, teacher features.
        student_feat: (B, D) tensor, student features.

    Returns:
        loss: Scalar tensor.
    """
    return F.mse_loss(student_feat, teacher_feat)


def compute_total_loss(
    student_image_features,
    student_text_features,
    teacher_image_features,
    teacher_text_features,
    student_logit_scale,
    student_logit_bias,
    teacher_logit_scale=15.6,
    teacher_logit_bias=-10.0,
    lambda_siglip=1.0,
    lambda_cmd=1.0,
    lambda_umd=1.0,
    temperature=3.0,
):
    """
    Computes the weighted total loss for TinySigLIP distillation.

    Args:
        student_image_features: (B, D) Student output features.
        student_text_features: (B, D) Student output features.
        teacher_image_features: (B, D) Cached teacher features.
        teacher_text_features: (B, D) Cached teacher features.
        student_logit_scale: Learnable parameter (scalar) from Student model.
        student_logit_bias: Learnable parameter (scalar) from Student model (for SigLIP).
        teacher_logit_scale: Fixed temperature scale for teacher model (for SigLIP).
        teacher_logit_bias: Fixed bias term for teacher model (for SigLIP).
        lambda_*: Weights for different loss components.
        temperature: Temperature for CMD loss.

    Returns:
        weighted_loss: The final scalar loss for backpropagation.
        loss_dict: Dictionary containing individual loss values for logging.
    """

    # --- 1. Feature Normalization ---
    # Normalize all features to the unit hypersphere (L2 norm).
    # Essential for calculating Cosine Similarity via dot product.
    s_img_norm = F.normalize(student_image_features, dim=-1)
    s_txt_norm = F.normalize(student_text_features, dim=-1)
    t_img_norm = F.normalize(teacher_image_features, dim=-1)
    t_txt_norm = F.normalize(teacher_text_features, dim=-1)

    # --- 2. Calculate Student Logits (Dynamic) ---
    # Formula: Scale * (Image @ Text.T) + Bias
    # The scale and bias are LEARNABLE parameters of the student.
    student_logits_per_image = student_logit_scale * s_img_norm @ s_txt_norm.T + student_logit_bias

    student_logits_per_text = student_logits_per_image.T

    # --- 3. Calculate Teacher Logits ---
    teacher_logits_per_image = teacher_logit_scale * t_img_norm @ t_txt_norm.T + teacher_logit_bias
    teacher_logits_per_text = teacher_logits_per_image.T

    # --- 4. Compute Loss Components ---

    # A. SigLIP Loss (Task Loss / Ground Truth Supervision)
    # Uses binary cross-entropy on the diagonal elements.
    loss_siglip = siglip_loss(student_logits_per_image, student_logits_per_text)

    # B. CMD Loss (Cross-Modal Distillation / Affinity)
    # Minimizes KL divergence between student and teacher similarity matrices.
    loss_cmd = (
        cross_modal_distillation_loss(teacher_logits_per_image, student_logits_per_image, temperature)
        + cross_modal_distillation_loss(teacher_logits_per_text, student_logits_per_text, temperature)
    ) / 2.0

    # C. UMD Loss (Uni-Modal Distillation / Feature Imitation)
    # Direct regression of embeddings (MSE).
    loss_umd_vision = feature_distillation_loss(t_img_norm, s_img_norm)
    loss_umd_text = feature_distillation_loss(t_txt_norm, s_txt_norm)
    loss_umd = (loss_umd_vision + loss_umd_text) / 2.0

    # --- 5. Weighted Sum ---
    loss_weighted = lambda_siglip * loss_siglip + lambda_cmd * loss_cmd + lambda_umd * loss_umd

    loss_dict = {
        "siglip": loss_siglip.item(),
        "cmd": loss_cmd.item(),
        "umd": loss_umd.item(),
        "weighted": loss_weighted.item(),
    }

    return loss_weighted, loss_dict
