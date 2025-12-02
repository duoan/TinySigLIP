"""
Image-text retrieval evaluation on full validation set.

This script evaluates the model on image-text retrieval tasks (e.g., COCO),
matching the evaluation protocol used in SigLIP 2 paper. This is different from
batch-internal evaluation during training, which only searches within a small batch.
"""

import argparse
import os
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from tinysiglip.coco_dataset import COCOCaptionDataset, collate_coco_batch
from tinysiglip.model import TinySiglipConfig, TinySiglipModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


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
            processor = AutoProcessor.from_pretrained(str(processor_path))
            return processor
        except Exception as e:
            print(f"Warning: Could not load processor from checkpoint: {e}")
            return None
    return None


def create_processor_from_config(config, device: str = "cuda"):
    """Create processor from config."""
    config_dict = config.get("config", config) if isinstance(config, dict) and "config" in config else config
    teacher_cfg = config_dict.get("teacher", {})
    student_cfg = config_dict.get("student", {})

    # Get teacher model name from config (preferred) or use student tokenizer name as fallback
    teacher_model_name = teacher_cfg.get("model_name", None)
    tokenizer_name = student_cfg.get("tokenizer_name", None)

    # Prefer teacher model name, fallback to tokenizer name, then default
    model_name = teacher_model_name or tokenizer_name or "google/siglip-base-patch16-224"

    try:
        processor = AutoProcessor.from_pretrained(model_name)
        return processor
    except Exception as e:
        print(f"Warning: Could not load processor from {model_name}: {e}")
        return None


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def measure_throughput_latency(
    model: TinySiglipModel,
    processor,  # AutoProcessor instance
    device: str = "cuda",
    num_warmup: int = 10,
    num_runs: int = 100,
    batch_size: int = 1,
):
    """
    Measure model throughput and latency.

    Args:
        model: TinySiglipModel instance
        processor: AutoProcessor instance
        device: Device to run on
        num_warmup: Number of warmup runs
        num_runs: Number of measurement runs
        batch_size: Batch size for measurement

    Returns:
        dict: Dictionary with throughput and latency metrics
    """
    model.eval()
    device_obj = torch.device(device)

    # Create dummy inputs
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device_obj)
    dummy_text = "a photo of a cat"
    # Process text - match the format used in evaluation
    text_inputs = processor(text=[dummy_text] * batch_size, return_tensors="pt", padding=True)
    dummy_text_ids = text_inputs["input_ids"].to(device_obj)
    # Note: evaluation code doesn't pass attention_mask, so we don't either

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_images, dummy_text_ids)

    # Synchronize if using CUDA
    if device_obj.type == "cuda":
        torch.cuda.synchronize()

    # Measure latency
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device_obj.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            _ = model(dummy_images, dummy_text_ids)
            if device_obj.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

    # Calculate metrics
    avg_latency_ms = sum(latencies) / len(latencies)
    min_latency_ms = min(latencies)
    max_latency_ms = max(latencies)
    p50_latency_ms = sorted(latencies)[len(latencies) // 2]
    p95_latency_ms = sorted(latencies)[int(len(latencies) * 0.95)]
    p99_latency_ms = sorted(latencies)[int(len(latencies) * 0.99)]

    # Throughput: samples per second
    throughput_samples_per_sec = (batch_size * 1000) / avg_latency_ms if avg_latency_ms > 0 else 0

    return {
        "latency_avg_ms": avg_latency_ms,
        "latency_min_ms": min_latency_ms,
        "latency_max_ms": max_latency_ms,
        "latency_p50_ms": p50_latency_ms,
        "latency_p95_ms": p95_latency_ms,
        "latency_p99_ms": p99_latency_ms,
        "throughput_samples_per_sec": throughput_samples_per_sec,
        "batch_size": batch_size,
    }


def evaluate_retrieval_coco(
    model: TinySiglipModel,
    processor,  # AutoProcessor instance
    dataset_path: str,  # Path to cached embeddings directory
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
    logit_scale: float | None = None,
    k_values=(1, 5, 10),
    max_samples: int | None = None,
    evaluate_teacher: bool = True,
):
    """
    Evaluate image-text retrieval on COCO validation set.

    Args:
        model: TinySiglipModel instance
        processor: AutoProcessor instance
        dataset_path: Path to cached embeddings directory
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        device: Device to run evaluation on
        logit_scale: Optional logit scale (temperature). If None, uses default 1.0
        k_values: K values for Recall@K
        max_samples: Maximum number of samples to evaluate (None for all)

    Returns:
        dict: Retrieval metrics. Keys:
            - i2t_recall@K / t2i_recall@K for student model
            - teacher_i2t_recall@K / teacher_t2i_recall@K if evaluate_teacher=True
    """
    # Create dataset
    print(f"Loading dataset from cache: {dataset_path}")
    # Get max_seq_len from tokenizer or use default
    max_seq_len = getattr(processor.tokenizer, "model_max_length", 64)
    if max_seq_len is None or max_seq_len == 1e30:  # Some tokenizers return this as default
        max_seq_len = 64

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

    # Optional: teacher features from cached embeddings (no teacher model needed)
    teacher_image_features_list = []
    teacher_text_features_list = []
    teacher_image_to_texts = {}  # Map unique teacher image index to list of text indices
    teacher_image_idx_map: dict[int, int] = {}  # image_id -> unique index

    print("Extracting features from dataset...")
    text_idx = 0
    last_image_feat = None
    unique_image_idx = -1
    similarity_threshold = 0.999  # Threshold for considering images as the same

    teacher_text_idx = 0

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

                # ===== Student features =====
                # Extract image and text features from student model
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

                # ===== Teacher features from cache (optional) =====
                if evaluate_teacher:
                    # Cached embeddings are per (image, caption) pair
                    # Use image_ids to deduplicate images across captions
                    image_id = batch["image_ids"][i]
                    # image_id comes from JSON / pickle, ensure it's int
                    image_id_int = int(image_id)

                    teacher_img = batch["teacher_image_embeds"][i].unsqueeze(0)
                    teacher_txt = batch["teacher_text_embeds"][i].unsqueeze(0)

                    teacher_img = F.normalize(teacher_img, dim=-1)
                    teacher_txt = F.normalize(teacher_txt, dim=-1)

                    # Create unique image index per image_id
                    if image_id_int not in teacher_image_idx_map:
                        t_unique_idx = len(teacher_image_features_list)
                        teacher_image_idx_map[image_id_int] = t_unique_idx
                        teacher_image_features_list.append(teacher_img.cpu())
                        teacher_image_to_texts[t_unique_idx] = []

                    t_img_idx = teacher_image_idx_map[image_id_int]
                    teacher_image_to_texts[t_img_idx].append(teacher_text_idx)
                    teacher_text_idx += 1

                    teacher_text_features_list.append(teacher_txt.cpu())

                if max_samples and text_idx >= max_samples:
                    break

            if max_samples and text_idx >= max_samples:
                break

    # Concatenate all student features
    all_image_features = torch.cat(all_image_features_list, dim=0).to(device)  # (N_unique_images, D)
    all_text_features = torch.cat(all_text_features_list, dim=0).to(device)  # (N_texts, D)

    print(f"\n[Student] Total unique images: {len(all_image_features)}")
    print(f"[Student] Total text captions: {len(all_text_features)}")
    print(f"[Student] Average captions per image: {len(all_text_features) / len(all_image_features):.2f}")

    # Use logit scale from checkpoint if available
    if logit_scale is None:
        logit_scale = 1.0

    # Compute similarity matrix for student model
    print("\nComputing similarity matrix for student model...")
    with torch.no_grad():
        # Image-to-text: (N_images, N_texts)
        logits_i2t = logit_scale * all_image_features @ all_text_features.T

        # Text-to-image: (N_texts, N_images)
        logits_t2i = logits_i2t.T

    # Compute Recall@K for image-to-text retrieval (student)
    print("\nComputing image-to-text retrieval metrics for student...")
    i2t_results = compute_recall_at_k_coco(logits_i2t, image_to_texts, k_values, "i2t")

    # Compute Recall@K for text-to-image retrieval (student)
    print("Computing text-to-image retrieval metrics for student...")
    t2i_results = compute_recall_at_k_coco(logits_t2i, image_to_texts, k_values, "t2i", reverse=True)

    results = {**i2t_results, **t2i_results}

    # ===== Optional: teacher metrics using cached embeddings =====
    if evaluate_teacher and teacher_image_features_list and teacher_text_features_list:
        teacher_image_features = torch.cat(teacher_image_features_list, dim=0).to(device)
        teacher_text_features = torch.cat(teacher_text_features_list, dim=0).to(device)

        print(f"\n[Teacher] Total unique images: {len(teacher_image_features)}")
        print(f"[Teacher] Total text captions: {len(teacher_text_features)}")
        print(f"[Teacher] Average captions per image: {len(teacher_text_features) / len(teacher_image_features):.2f}")

        # For teacher we use pure cosine similarity (logit_scale = 1.0),
        # matching the typical evaluation for frozen encoders.
        print("\nComputing similarity matrix for teacher model (cached embeddings)...")
        with torch.no_grad():
            teacher_logits_i2t = teacher_image_features @ teacher_text_features.T
            teacher_logits_t2i = teacher_logits_i2t.T

        print("\nComputing image-to-text retrieval metrics for teacher...")
        teacher_i2t_results = compute_recall_at_k_coco(
            teacher_logits_i2t, teacher_image_to_texts, k_values, "teacher_i2t"
        )

        print("Computing text-to-image retrieval metrics for teacher...")
        teacher_t2i_results = compute_recall_at_k_coco(
            teacher_logits_t2i, teacher_image_to_texts, k_values, "teacher_t2i", reverse=True
        )

        results.update(teacher_i2t_results)
        results.update(teacher_t2i_results)

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


def auto_detect_dataset_path(checkpoint, split: str, cache_dir: str | None = None) -> str | None:
    """
    Auto-detect dataset path from checkpoint config and cache directory.

    Args:
        checkpoint: Checkpoint dictionary containing config
        split: Dataset split ('val' or 'test')
        cache_dir: Base cache directory (default: "data/coco/cache")

    Returns:
        Detected dataset path or None if not found
    """
    project_root = Path(__file__).parent.resolve()

    # Default cache directory
    if cache_dir is None:
        cache_dir = "data/coco/cache"

    cache_base_dir = Path(cache_dir)
    if not cache_base_dir.is_absolute():
        cache_base_dir = project_root / cache_base_dir

    # Get teacher model name from checkpoint config
    config = checkpoint.get("config", {})
    teacher_cfg = config.get("teacher", {})
    teacher_model_name = teacher_cfg.get("model_name", "google/siglip2-base-patch16-224")

    # Sanitize model name for filesystem
    safe_model_name = re.sub(r"[^\w\-_]", "_", teacher_model_name).strip("_")

    # Try different dataset sizes (tiny, medium, large)
    # For each size, try the requested split first, then fall back to other common splits
    splits_to_try = [split] + [s for s in ["train", "val", "test"] if s != split]

    for dataset_size_name in ["tiny", "medium", "large"]:
        for split_name in splits_to_try:
            cache_path_candidate = cache_base_dir / safe_model_name / dataset_size_name / split_name
            metadata_file = cache_path_candidate / "metadata.json"
            if metadata_file.exists():
                dataset_path = str(cache_path_candidate)
                if split_name != split:
                    print(
                        f"⚠ Warning: Requested split '{split}' but found cache for "
                        f"'{split_name}'. Using: {dataset_path}"
                    )
                else:
                    print(f"✓ Auto-detected dataset path: {dataset_path}")
                return dataset_path

    return None


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
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help=(
            "Path to cached embeddings directory. If None, will try to auto-detect "
            "based on checkpoint config and split."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=(
            "Base cache directory for auto-detection (default: data/coco/cache). "
            "Only used if --dataset-path is not provided."
        ),
    )
    parser.add_argument(
        "--skip-teacher",
        action="store_true",
        help="Skip evaluation of teacher (cached) embeddings and only evaluate the student model",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="tinysiglip-eval",
        help="WandB project name (default: tinysiglip-eval)",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="WandB run name (default: auto-generated from checkpoint path)",
    )
    parser.add_argument(
        "--skip-wandb",
        action="store_true",
        help="Skip WandB logging",
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
        # Try to get teacher model name from config
        teacher_cfg = config.get("teacher", {})
        teacher_model_name = teacher_cfg.get("model_name", None)

        if teacher_model_name:
            try:
                processor = AutoProcessor.from_pretrained(teacher_model_name)
                print(f"✓ Loaded processor from teacher model: {teacher_model_name}")
            except Exception as e:
                print(f"Warning: Could not load processor from {teacher_model_name}: {e}")
                processor = create_processor_from_config(config, device=args.device)
        else:
            processor = create_processor_from_config(config, device=args.device)

    if processor is None:
        raise ValueError("Could not load or create processor. Please check checkpoint and config.")

    # Initialize WandB
    wandb_enabled = WANDB_AVAILABLE and not args.skip_wandb
    if wandb_enabled:
        checkpoint_name = Path(args.resume).stem
        run_name = args.wandb_name or f"eval_{checkpoint_name}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "checkpoint": args.resume,
                "split": args.split,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "device": args.device,
                "logit_scale": logit_scale,
                "max_samples": args.max_samples,
                "evaluate_teacher": not args.skip_teacher,
            },
        )
        print(f"✓ WandB initialized: {wandb.run.url if wandb.run else 'N/A'}")

    # Calculate model parameters
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n=== Model Parameters ===")
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print("=" * 30)

    # Measure throughput and latency
    print("\n=== Measuring Throughput and Latency ===")
    perf_metrics = measure_throughput_latency(
        model=model,
        processor=processor,
        device=args.device,
        num_warmup=10,
        num_runs=100,
        batch_size=1,
    )
    print(f"Average latency: {perf_metrics['latency_avg_ms']:.2f} ms")
    print(f"P50 latency: {perf_metrics['latency_p50_ms']:.2f} ms")
    print(f"P95 latency: {perf_metrics['latency_p95_ms']:.2f} ms")
    print(f"Throughput: {perf_metrics['throughput_samples_per_sec']:.2f} samples/sec")
    print("=" * 30)

    # Determine dataset path
    dataset_path = args.dataset_path
    if dataset_path is None:
        # Try to auto-detect from checkpoint config
        print("Auto-detecting dataset path from checkpoint config...")
        dataset_path = auto_detect_dataset_path(checkpoint, args.split, args.cache_dir)
        if dataset_path is None:
            raise ValueError(
                f"Could not auto-detect dataset path for split '{args.split}'. "
                "Please specify --dataset-path explicitly. "
                "Expected format: data/coco/cache/model_name/dataset_size/split"
            )
    else:
        # Convert to absolute path if relative
        if not Path(dataset_path).is_absolute():
            project_root = Path(__file__).parent.resolve()
            dataset_path = str(project_root / dataset_path)

    # Validate dataset path
    metadata_file = Path(dataset_path) / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Dataset cache not found: {dataset_path}\nPlease run prepare_data.py first to generate the cache."
        )

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating on COCO Caption dataset")
    print("=" * 60 + "\n")

    results = evaluate_retrieval_coco(
        model=model,
        processor=processor,
        dataset_path=dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        logit_scale=logit_scale,
        max_samples=args.max_samples,
        evaluate_teacher=not args.skip_teacher,
    )

    if results:
        print("\n" + "=" * 60)
        print("COCO Image-Text Retrieval Results:")
        print("=" * 60)

        # Teacher results (if available)
        has_teacher = any(key.startswith("teacher_") for key in results.keys())
        if has_teacher:
            print("Teacher (cached SigLIP) Image-to-Text Retrieval:")
            for k in [1, 5, 10]:
                key = f"teacher_i2t_recall@{k}"
                if key in results:
                    print(f"  Recall@{k}: {results[key]:.2f}%")
            print("\nTeacher (cached SigLIP) Text-to-Image Retrieval:")
            for k in [1, 5, 10]:
                key = f"teacher_t2i_recall@{k}"
                if key in results:
                    print(f"  Recall@{k}: {results[key]:.2f}%")
            print("\n" + "-" * 60)

        # Student (TinySigLIP) results
        print("Student (TinySigLIP) Image-to-Text Retrieval:")
        for k in [1, 5, 10]:
            key = f"i2t_recall@{k}"
            if key in results:
                print(f"  Recall@{k}: {results[key]:.2f}%")

        print("\nStudent (TinySigLIP) Text-to-Image Retrieval:")
        for k in [1, 5, 10]:
            key = f"t2i_recall@{k}"
            if key in results:
                print(f"  Recall@{k}: {results[key]:.2f}%")

        print("=" * 60 + "\n")
        print("Evaluation completed successfully!")

        # Log to WandB tables
        if wandb_enabled:
            # 1. Create vertical summary table (transposed format)
            summary_table_data = [
                ["Checkpoint", Path(args.resume).name],
                ["Split", args.split],
                ["Total Params (M)", f"{total_params / 1e6:.2f}"],
                ["Trainable Params (M)", f"{trainable_params / 1e6:.2f}"],
                ["Throughput (samples/sec)", f"{perf_metrics['throughput_samples_per_sec']:.2f}"],
                ["Latency Avg (ms)", f"{perf_metrics['latency_avg_ms']:.2f}"],
                ["Latency P50 (ms)", f"{perf_metrics['latency_p50_ms']:.2f}"],
                ["Latency P95 (ms)", f"{perf_metrics['latency_p95_ms']:.2f}"],
                ["Latency P99 (ms)", f"{perf_metrics['latency_p99_ms']:.2f}"],
            ]
            summary_table = wandb.Table(columns=["Metric", "Value"], data=summary_table_data)
            wandb.log({"summary_table": summary_table})

            # 2. Create teacher vs student retrieval recall comparison table
            has_teacher = any(key.startswith("teacher_") for key in results.keys())
            recall_comparison_data = []

            # Image-to-Text Retrieval section
            recall_comparison_data.append(["Image-to-Text Retrieval", "", "", ""])
            for k in [1, 5, 10]:
                i2t_key = f"i2t_recall@{k}"
                student_val = results.get(i2t_key, None)
                teacher_val = results.get(f"teacher_{i2t_key}", None) if has_teacher else None

                student_str = f"{student_val:.2f}%" if student_val is not None else "N/A"
                teacher_str = f"{teacher_val:.2f}%" if teacher_val is not None else "N/A"

                # Calculate gap if both available
                if student_val is not None and teacher_val is not None:
                    gap = student_val - teacher_val
                    gap_str = f"{gap:+.2f}%"
                else:
                    gap_str = "N/A"

                recall_comparison_data.append([f"Recall@{k}", student_str, teacher_str, gap_str])

            # Text-to-Image Retrieval section
            recall_comparison_data.append(["Text-to-Image Retrieval", "", "", ""])
            for k in [1, 5, 10]:
                t2i_key = f"t2i_recall@{k}"
                student_val = results.get(t2i_key, None)
                teacher_val = results.get(f"teacher_{t2i_key}", None) if has_teacher else None

                student_str = f"{student_val:.2f}%" if student_val is not None else "N/A"
                teacher_str = f"{teacher_val:.2f}%" if teacher_val is not None else "N/A"

                # Calculate gap if both available
                if student_val is not None and teacher_val is not None:
                    gap = student_val - teacher_val
                    gap_str = f"{gap:+.2f}%"
                else:
                    gap_str = "N/A"

                recall_comparison_data.append([f"Recall@{k}", student_str, teacher_str, gap_str])

            # Create comparison table
            recall_comparison_table = wandb.Table(
                columns=["Metric", "Student", "Teacher", "Gap (Student - Teacher)"],
                data=recall_comparison_data,
            )
            wandb.log({"retrieval_recall_comparison": recall_comparison_table})

            # Also log as summary metrics for easy comparison
            wandb.summary.update(
                {
                    "model/total_params": total_params,
                    "model/total_params_M": total_params / 1e6,
                    "model/trainable_params": trainable_params,
                    "model/trainable_params_M": trainable_params / 1e6,
                    "performance/throughput_samples_per_sec": perf_metrics["throughput_samples_per_sec"],
                    "performance/latency_avg_ms": perf_metrics["latency_avg_ms"],
                    "performance/latency_p50_ms": perf_metrics["latency_p50_ms"],
                    "performance/latency_p95_ms": perf_metrics["latency_p95_ms"],
                    "performance/latency_p99_ms": perf_metrics["latency_p99_ms"],
                }
            )

            # Add retrieval metrics to summary
            for key, value in results.items():
                wandb.summary[f"retrieval/{key}"] = value

            # Upload checkpoint to WandB as artifact
            checkpoint_path = Path(args.resume)
            if checkpoint_path.exists():
                print("\n=== Uploading checkpoint to WandB ===")
                try:
                    # Create artifact with descriptive name
                    checkpoint_name = checkpoint_path.stem
                    artifact_name = f"tinysiglip-checkpoint-{checkpoint_name}"
                    artifact = wandb.Artifact(
                        artifact_name,
                        type="model",
                        description=f"TinySigLIP model checkpoint: {checkpoint_name}",
                        metadata={
                            "checkpoint_path": str(checkpoint_path),
                            "total_params_M": total_params / 1e6,
                            "trainable_params_M": trainable_params / 1e6,
                            "split": args.split,
                            "logit_scale": logit_scale,
                        },
                    )

                    # Add checkpoint file
                    artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)
                    print(f"  Added checkpoint file: {checkpoint_path.name}")

                    # Also add processor directory if it exists
                    checkpoint_dir = checkpoint_path.parent
                    processor_dir = checkpoint_dir / "processor"
                    if processor_dir.exists() and processor_dir.is_dir():
                        artifact.add_dir(str(processor_dir), name="processor")
                        print("  Added processor directory: processor/")

                    # Add model config if it exists
                    model_config_file = checkpoint_dir / "model_config.json"
                    if model_config_file.exists():
                        artifact.add_file(str(model_config_file), name="model_config.json")
                        print("  Added model config: model_config.json")

                    # Add training config if it exists
                    config_file = checkpoint_dir / "config.yaml"
                    if config_file.exists():
                        artifact.add_file(str(config_file), name="config.yaml")
                        print("  Added training config: config.yaml")

                    # Log artifact
                    wandb.log_artifact(artifact)
                    print(f"✓ Checkpoint uploaded to WandB as artifact: {artifact_name}")
                except Exception as e:
                    print(f"⚠ Warning: Failed to upload checkpoint to WandB: {e}")

            print("\n✓ Results logged to WandB")
            if wandb.run:
                run_url = wandb.run.url if hasattr(wandb.run, "url") else None
                wandb.finish()
                if run_url:
                    print(f"✓ WandB run completed: {run_url}")
                else:
                    print("✓ WandB run completed")


if __name__ == "__main__":
    main()
