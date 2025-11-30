"""
Token Embedding Layer Distillation utilities.

This module provides functions to create token mappings between teacher and student
vocabularies and transfer embedding weights.
"""

import torch


def create_token_mapping(
    teacher_tokenizer,
    student_tokenizer,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create token mapping using actual tokenizers to find shared tokens.

    This function compares the vocabularies of teacher and student tokenizers
    to find tokens that exist in both, and maps their indices accordingly.

    Args:
        teacher_tokenizer: Teacher model's tokenizer
        student_tokenizer: Student model's tokenizer
        verbose: Whether to print mapping statistics

    Returns:
        Tuple of (shared_token_indices_student, shared_token_indices_teacher)
        Both are LongTensors containing the indices of shared tokens
    """
    # Get vocabularies
    teacher_vocab = teacher_tokenizer.get_vocab()
    student_vocab = student_tokenizer.get_vocab()

    # Find shared tokens (tokens that exist in both vocabularies)
    shared_tokens = []
    shared_student_indices = []
    shared_teacher_indices = []

    for token, student_idx in student_vocab.items():
        if token in teacher_vocab:
            teacher_idx = teacher_vocab[token]
            shared_tokens.append(token)
            shared_student_indices.append(student_idx)
            shared_teacher_indices.append(teacher_idx)

    if verbose:
        print("\n=== Token Mapping (Real Tokenizers) ===")
        print(f"Teacher vocab size: {len(teacher_vocab)}")
        print(f"Student vocab size: {len(student_vocab)}")
        print(f"Shared tokens found: {len(shared_tokens)}")
        if len(shared_tokens) > 0:
            overlap_ratio = len(shared_tokens) / len(student_vocab)
            print(f"Overlap ratio: {overlap_ratio:.2%}")
            # Show some examples
            print(f"Example shared tokens (first 10): {shared_tokens[:10]}")
        print("==========================================\n")

    return (
        torch.tensor(shared_student_indices, dtype=torch.long),
        torch.tensor(shared_teacher_indices, dtype=torch.long),
    )


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


def transfer_embedding_weights(
    student_embedding_layer: torch.nn.Embedding,
    teacher_embedding_layer: torch.nn.Embedding,
    shared_student_indices: torch.Tensor,
    shared_teacher_indices: torch.Tensor,
    verbose: bool = True,
) -> int:
    """
    Transfer embedding weights using pre-computed token mappings.

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

        # Validate indices
        if student_id >= student_weight.shape[0] or teacher_id >= teacher_weight.shape[0]:
            if verbose and transferred_count == 0:  # Only print warning once
                print(f"Warning: Skipping invalid index pair (student_id={student_id}, teacher_id={teacher_id})")
            continue

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
        print("\n=== Embedding Weight Transfer ===")
        print(f"Transferred {transferred_count} token embeddings")
        print("==================================\n")

    return transferred_count
