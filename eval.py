"""
Image-text retrieval evaluation on full validation set.

This script evaluates the model on image-text retrieval tasks (e.g., COCO),
matching the evaluation protocol used in SigLIP 2 paper. This is different from
batch-internal evaluation during training, which only searches within a small batch.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from tinysiglip.coco_dataset import COCOCaptionDataset, collate_coco_batch
from tinysiglip.model import TinySiglipConfig, TinySiglipModel
from tinysiglip.processor import TinySiglipProcessor


def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def create_model_from_checkpoint(checkpoint, device: str = "cuda"):
    """Create and load model from checkpoint."""
    # Try to load model config from checkpoint (new format)
    if "model_config" in checkpoint:
        model_config = TinySiglipConfig.from_dict(checkpoint["model_config"])
    else:
        # Fallback to old format: extract from config dict
        config = checkpoint.get("config", {})
        student_cfg = config.get("student", {})
        training_cfg = config.get("training", {})
        model_config = TinySiglipConfig(
            vision_model_name=student_cfg.get("vision_model_name", "vit_tiny_patch16_224"),
            text_vocab_size=student_cfg.get("vocab_size", 32000),
            text_seq_len=training_cfg.get("text_seq_len", 64),
            text_dim=student_cfg.get("text_dim", 384),
            text_layers=student_cfg.get("text_layers", 4),
            text_nhead=student_cfg.get("text_nhead", 8),
            text_ff_dim_multiplier=student_cfg.get("text_ff_dim_multiplier", 4),
            projection_dim=student_cfg.get("projection_dim", 384),
        )

    # Create model from config
    model = TinySiglipModel(config=model_config).to(device)

    # Load weights
    model.load_state_dict(checkpoint["student_model"])
    model.eval()

    return model


def load_processor_from_checkpoint(checkpoint_dir: Path):
    """Load processor from checkpoint directory."""
    processor_path = checkpoint_dir / "processor"
    if processor_path.exists():
        try:
            processor = TinySiglipProcessor.from_pretrained(str(processor_path))
            return processor
        except Exception as e:
            print(f"Warning: Could not load processor from checkpoint: {e}")
            return None
    return None


def create_processor_from_config(config, device: str = "cuda"):
    """Create processor from config."""
    from transformers import AutoImageProcessor, AutoTokenizer

    student_cfg = config.get("student", {})
    training_cfg = config.get("training", {})

    tokenizer_name = student_cfg.get("tokenizer_name", "google/siglip-base-patch16-224")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Warning: Could not load tokenizer {tokenizer_name}: {e}")
        return None

    try:
        image_processor = AutoImageProcessor.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Warning: Could not load image processor: {e}")
        image_processor = None

    processor = TinySiglipProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_size=training_cfg.get("image_size", 224),
        max_seq_len=training_cfg.get("text_seq_len", 64),
        use_augmentation=False,  # No augmentation for evaluation
    )

    return processor


def evaluate_retrieval_coco(
    model: TinySiglipModel,
    processor: TinySiglipProcessor,
    dataset_path: str,  # Path to cached embeddings directory
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
    logit_scale: float | None = None,
    k_values=(1, 5, 10),
    max_samples: int | None = None,
):
    """
    Evaluate image-text retrieval on COCO validation set.

    Args:
        model: TinySiglipModel instance
        processor: TinySiglipProcessor instance
        split: Dataset split ('val' or 'test')
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        device: Device to run evaluation on
        logit_scale: Optional logit scale (temperature). If None, uses default 1.0
        cache_dir: Directory to cache the dataset
        k_values: K values for Recall@K
        max_samples: Maximum number of samples to evaluate (None for all)

    Returns:
        dict: Retrieval metrics
    """
    # Create dataset
    print(f"Loading dataset from cache: {dataset_path}")
    max_seq_len = processor.max_seq_len if hasattr(processor, "max_seq_len") else 64

    dataset = COCOCaptionDataset(
        dataset_path=dataset_path,
        processor=processor,
        max_seq_len=max_seq_len,
        verbose=True,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_coco_batch,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
    )

    model.eval()

    # Extract all image and text features
    # COCO dataset yields one sample per (image, caption) pair
    # Same image appears multiple times (once per caption)
    # We need to deduplicate images and track which captions belong to which image
    all_image_features_list = []
    all_text_features_list = []
    image_to_texts = {}  # Map unique image index to list of text indices

    print("Extracting features from dataset...")
    text_idx = 0
    last_image_feat = None
    unique_image_idx = -1
    similarity_threshold = 0.999  # Threshold for considering images as the same

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            if max_samples and text_idx >= max_samples:
                break

            images = batch["student_images"].to(device)
            text_ids_batch = batch["student_text_ids"].to(device)
            batch_size_actual = images.size(0)

            # Process each sample in the batch
            for i in range(batch_size_actual):
                image = images[i : i + 1]
                text_id = text_ids_batch[i : i + 1]

                # Extract image and text features
                image_feat, text_feat = model(image, text_id)
                image_feat = F.normalize(image_feat, dim=-1)
                text_feat = F.normalize(text_feat, dim=-1)
                all_text_features_list.append(text_feat.cpu())

                # Check if this is a new image or same image with different caption
                # We assume same image appears consecutively, so compare with last image
                is_new_image = True
                if last_image_feat is not None:
                    # Compare with last image feature
                    similarity = (image_feat @ last_image_feat.T).item()
                    if similarity > similarity_threshold:
                        # Same image, reuse the same unique image index
                        is_new_image = False

                if is_new_image:
                    # New unique image
                    unique_image_idx += 1
                    all_image_features_list.append(image_feat.cpu())
                    image_to_texts[unique_image_idx] = []
                    last_image_feat = image_feat

                # Map this text caption to the unique image
                image_to_texts[unique_image_idx].append(text_idx)
                text_idx += 1

                if max_samples and text_idx >= max_samples:
                    break

            if max_samples and text_idx >= max_samples:
                break

    # Concatenate all features
    all_image_features = torch.cat(all_image_features_list, dim=0).to(device)  # (N_unique_images, D)
    all_text_features = torch.cat(all_text_features_list, dim=0).to(device)  # (N_texts, D)

    print(f"\nTotal unique images: {len(all_image_features)}")
    print(f"Total text captions: {len(all_text_features)}")
    print(f"Average captions per image: {len(all_text_features) / len(all_image_features):.2f}")

    # Use logit scale from checkpoint if available
    if logit_scale is None:
        logit_scale = 1.0

    # Compute similarity matrix
    print("\nComputing similarity matrix...")
    with torch.no_grad():
        # Image-to-text: (N_images, N_texts)
        logits_i2t = logit_scale * all_image_features @ all_text_features.T

        # Text-to-image: (N_texts, N_images)
        logits_t2i = logits_i2t.T

    # Compute Recall@K for image-to-text retrieval
    print("\nComputing image-to-text retrieval metrics...")
    i2t_results = compute_recall_at_k_coco(logits_i2t, image_to_texts, k_values, "i2t")

    # Compute Recall@K for text-to-image retrieval
    print("Computing text-to-image retrieval metrics...")
    t2i_results = compute_recall_at_k_coco(logits_t2i, image_to_texts, k_values, "t2i", reverse=True)

    results = {**i2t_results, **t2i_results}

    return results


def compute_recall_at_k_coco(logits, image_to_texts, k_values, prefix="", reverse=False):
    """
    Compute Recall@K for COCO retrieval.

    Args:
        logits: (N_queries, N_candidates) similarity matrix
        image_to_texts: Dict mapping image index to list of text indices (or reverse)
        k_values: K values for Recall@K
        prefix: Prefix for metric names
        reverse: If True, reverse the mapping (for text-to-image)

    Returns:
        dict: Recall@K metrics
    """
    results = {}
    correct_at_k = dict.fromkeys(k_values, 0)
    total_queries = 0

    with torch.no_grad():
        _, top_indices = torch.topk(logits, k=max(k_values), dim=1)

        for query_idx in range(logits.size(0)):
            # Find correct candidate indices
            if reverse:
                # Text-to-image: text_idx -> image indices
                # Find which image this text belongs to
                correct_indices = set()
                for img_idx, text_indices in image_to_texts.items():
                    if query_idx in text_indices:
                        correct_indices.add(img_idx)
            else:
                # Image-to-text: image_idx -> text indices
                correct_indices = set(image_to_texts.get(query_idx, []))

            if correct_indices:
                # Check if any correct candidate is in top-k
                for k in k_values:
                    top_k = set(top_indices[query_idx, :k].cpu().tolist())
                    if top_k & correct_indices:  # Intersection
                        correct_at_k[k] += 1

                total_queries += 1

    # Compute recall
    for k in k_values:
        recall = correct_at_k[k] / total_queries if total_queries > 0 else 0.0
        results[f"{prefix}_recall@{k}"] = recall * 100.0  # Convert to percentage

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate TinySigLIP on COCO image-text retrieval")
    parser.add_argument(
        "--resume",
        type=str,
        required=True,
        help="Path to checkpoint file (.pt)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate on (default: val)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--logit-scale",
        type=float,
        default=None,
        help="Logit scale (temperature). If None, uses checkpoint value or default 1.0",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: None for all)",
    )

    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from {args.resume}...")
    checkpoint = load_checkpoint(args.resume, device=args.device)

    # Get logit scale from checkpoint if available
    logit_scale = args.logit_scale
    if logit_scale is None and "logit_scale" in checkpoint:
        logit_scale_val = checkpoint["logit_scale"]
        logit_scale = logit_scale_val.item() if torch.is_tensor(logit_scale_val) else logit_scale_val
        print(f"Using logit scale from checkpoint: {logit_scale:.4f}")
    elif logit_scale is None:
        logit_scale = 1.0
        print(f"Using default logit scale: {logit_scale}")

    # Create model
    print("Creating model from checkpoint...")
    model = create_model_from_checkpoint(checkpoint, device=args.device)

    # Load or create processor
    checkpoint_dir = Path(args.resume).parent
    processor = load_processor_from_checkpoint(checkpoint_dir)

    if processor is None:
        print("Creating processor from config...")
        config = checkpoint.get("config", {})
        processor = create_processor_from_config(config, device=args.device)

    if processor is None:
        raise ValueError("Could not load or create processor. Please check checkpoint and config.")

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating on COCO Caption dataset")
    print("=" * 60 + "\n")

    results = evaluate_retrieval_coco(
        model=model,
        processor=processor,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        logit_scale=logit_scale,
        max_samples=args.max_samples,
    )

    if results:
        print("\n" + "=" * 60)
        print("COCO Image-Text Retrieval Results:")
        print("=" * 60)
        print("Image-to-Text Retrieval:")
        for k in [1, 5, 10]:
            key = f"i2t_recall@{k}"
            if key in results:
                print(f"  Recall@{k}: {results[key]:.2f}%")
        print("\nText-to-Image Retrieval:")
        for k in [1, 5, 10]:
            key = f"t2i_recall@{k}"
            if key in results:
                print(f"  Recall@{k}: {results[key]:.2f}%")
        print("=" * 60 + "\n")
        print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
