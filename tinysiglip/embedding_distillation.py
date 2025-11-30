"""
Token Embedding Layer Distillation utilities.

This module provides functions to create token mappings between teacher and student
vocabularies and transfer embedding weights.
"""

import torch


def create_dummy_token_mapping(
    student_vocab_size: int, teacher_vocab_size: int, overlap_ratio: float = 0.8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a dummy token mapping for demonstration purposes.

    In real applications, you should use actual tokenizers to find shared tokens:
    1. Load both teacher and student tokenizers
    2. Find tokens that exist in both vocabularies
    3. Map their indices accordingly

    Args:
        student_vocab_size: Size of student vocabulary
        teacher_vocab_size: Size of teacher vocabulary
        overlap_ratio: Ratio of student vocab that overlaps with teacher (0-1)

    Returns:
        Tuple of (shared_token_indices_student, shared_token_indices_teacher)
        Both are LongTensors containing the indices of shared tokens
    """
    # For dummy data, assume the first N tokens of student vocab
    # map to some subset of teacher vocab
    num_shared = int(student_vocab_size * overlap_ratio)

    # Student indices: first N tokens (assuming they're shared)
    shared_token_indices_student = torch.arange(num_shared, dtype=torch.long)

    # Teacher indices: map to a subset of teacher vocab
    # In real case, this would be determined by tokenizer comparison
    # For now, we map student token i -> teacher token i * step
    step = max(1, teacher_vocab_size // num_shared)
    shared_token_indices_teacher = torch.arange(0, num_shared * step, step, dtype=torch.long)
    # Clamp to valid range
    shared_token_indices_teacher = torch.clamp(shared_token_indices_teacher, max=teacher_vocab_size - 1)

    return shared_token_indices_student, shared_token_indices_teacher


def transfer_embedding_weights_dummy(
    student_embedding_layer: torch.nn.Embedding,
    teacher_embedding_layer: torch.nn.Embedding,
    shared_student_indices: torch.Tensor,
    shared_teacher_indices: torch.Tensor,
    verbose: bool = True,
) -> int:
    """
    Transfer embedding weights using pre-computed token mappings (for dummy data).

    Args:
        student_embedding_layer: Student's embedding layer
        teacher_embedding_layer: Teacher's embedding layer
        shared_student_indices: Pre-computed student token indices
        shared_teacher_indices: Pre-computed teacher token indices
        verbose: Whether to print transfer statistics

    Returns:
        Number of tokens successfully transferred
    """
    student_weight = student_embedding_layer.weight.data
    teacher_weight = teacher_embedding_layer.weight.data

    transferred_count = 0

    for i in range(len(shared_student_indices)):
        student_id = int(shared_student_indices[i].item())
        teacher_id = int(shared_teacher_indices[i].item())

        teacher_emb = teacher_weight[teacher_id]

        # Handle dimension mismatch
        if student_weight.shape[1] == teacher_weight.shape[1]:
            student_weight[student_id].copy_(teacher_emb)
        elif teacher_weight.shape[1] > student_weight.shape[1]:
            student_weight[student_id].copy_(teacher_emb[: student_weight.shape[1]])
        else:
            student_weight[student_id, : teacher_weight.shape[1]].copy_(teacher_emb)
            student_weight[student_id, teacher_weight.shape[1] :].zero_()

        transferred_count += 1

    if verbose:
        print("\n=== Embedding Weight Transfer (Dummy) ===")
        print(f"Transferred {transferred_count} token embeddings")
        print("==========================================\n")

    return transferred_count
