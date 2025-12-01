"""
Simplified training script for SigLIP distillation with Hydra configuration.
"""

import math
import os
import time
from pathlib import Path
from typing import Any, cast

import hydra
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, get_cosine_schedule_with_warmup

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

try:
    from fvcore.nn import flop_count  # type: ignore[import-untyped]

    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False

from tinysiglip.coco_dataset import COCOCaptionDataset, collate_coco_batch
from tinysiglip.loss import compute_total_loss
from tinysiglip.metrics import compute_evaluation_metrics
from tinysiglip.model import TinySiglipConfig, TinySiglipModel


def setup_distributed():
    """Initialize distributed training environment."""
    # Check if running in distributed mode
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

        # Initialize process group
        dist.init_process_group(backend="nccl")

        # Set device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        return True, rank, world_size, local_rank, device
    else:
        # Single GPU or CPU mode
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # macOS Metal Performance Shaders (Apple Silicon)
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        return False, 0, 1, 0, device


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if current process is the main process (rank 0)."""
    return rank == 0


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: int) -> str:
    """Format number with appropriate unit (M for millions, K for thousands)."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def compute_flops(
    model: nn.Module,
    image_size: int,
    text_seq_len: int,
    device: torch.device,
) -> dict[str, float | None | str]:
    """
    Compute FLOPs (Floating Point Operations) for the model.

    Args:
        model: The model to compute FLOPs for
        image_size: Image size (assumed square)
        text_seq_len: Text sequence length
        device: Device to run computation on

    Returns:
        dict: Dictionary with FLOPs metrics
    """
    if FVCORE_AVAILABLE:
        try:
            # Create dummy inputs
            dummy_images = torch.randn(1, 3, image_size, image_size).to(device)
            dummy_text_ids = torch.randint(1, 1000, (1, text_seq_len)).to(device)

            # Count FLOPs
            flops_dict, _ = flop_count(model, (dummy_images, dummy_text_ids))

            # fvcore returns a dict where values might be FlopCountMode objects or numbers
            # Convert all values to numbers
            total_flops = 0.0
            for value in flops_dict.values():
                if isinstance(value, (int, float)):
                    total_flops += float(value)
                elif hasattr(value, "total"):
                    # FlopCountMode object - check if total is callable
                    total_method = getattr(value, "total", None)
                    if callable(total_method):
                        total_flops += float(total_method())  # type: ignore[arg-type]
                    else:
                        # Try to use as number
                        try:
                            total_flops += float(value)  # type: ignore[arg-type]
                        except (ValueError, TypeError):
                            pass
                else:
                    # Try to convert to number
                    try:
                        total_flops += float(value)  # type: ignore[arg-type]
                    except (ValueError, TypeError):
                        pass

            # If total_flops is still very small, fvcore might not have captured everything
            # Fall back to estimation if result seems unreasonable (< 1M FLOPs)
            if total_flops < 1e6:
                # Use estimation instead
                total_params = sum(p.numel() for p in model.parameters())
                # Rough estimate: for vision transformer + text transformer
                # Vision: approximately 2 * image_size^2 * num_patches * hidden_dim operations
                # Text: approximately 2 * text_seq_len^2 * hidden_dim operations
                # This is a simplified estimate
                vision_flops_estimate = image_size * image_size * 3 * 2  # Input processing
                text_flops_estimate = text_seq_len * text_seq_len * 2  # Attention
                # Add parameter-based estimate
                param_flops = total_params * 2  # Rough: 2 FLOPs per parameter
                total_flops = vision_flops_estimate + text_flops_estimate + param_flops

            return {
                "total_flops": total_flops,
                "total_flops_g": total_flops / 1e9,  # GFLOPs
                "flops_per_sample": total_flops,
            }
        except Exception as e:
            print(f"Warning: Could not compute FLOPs with fvcore: {e}")
            # Fall back to estimation
            total_params = sum(p.numel() for p in model.parameters())
            vision_flops_estimate = image_size * image_size * 3 * 2
            text_flops_estimate = text_seq_len * text_seq_len * 2
            param_flops = total_params * 2
            total_flops_estimate = vision_flops_estimate + text_flops_estimate + param_flops
            return {
                "total_flops": total_flops_estimate,
                "total_flops_g": total_flops_estimate / 1e9,
                "flops_per_sample": total_flops_estimate,
                "note": "Estimated (fvcore failed)",
            }
    else:
        # Simple estimation based on model parameters and operations
        # This is a rough estimate
        total_params = sum(p.numel() for p in model.parameters())
        # Rough estimate: 2 FLOPs per parameter per forward pass
        # For vision: image_size^2 * 3 * 2 (rough estimate)
        # For text: text_seq_len * vocab_size * 2 (rough estimate)
        vision_flops_estimate = image_size * image_size * 3 * 2 * total_params * 0.5  # Rough estimate
        text_flops_estimate = text_seq_len * 1000 * 2 * total_params * 0.5  # Rough estimate
        total_flops_estimate = vision_flops_estimate + text_flops_estimate

        return {
            "total_flops": total_flops_estimate,
            "total_flops_g": total_flops_estimate / 1e9,
            "flops_per_sample": total_flops_estimate,
            "note": "Estimated (install fvcore for accurate FLOPs)",
        }


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function with Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    # Initialize distributed training
    use_distributed, rank, world_size, local_rank, device = setup_distributed()

    if use_distributed:
        if is_main_process(rank):
            print(f"Initialized distributed training: world_size={world_size}, rank={rank}, local_rank={local_rank}")
    else:
        print(f"Using device: {device}")

    # Extract configuration values
    BATCH_SIZE = cfg.training.batch_size
    IMAGE_SIZE = cfg.training.image_size
    TEXT_SEQ_LEN = cfg.training.text_seq_len
    NUM_EPOCHS = cfg.training.num_epochs
    LEARNING_RATE = cfg.training.learning_rate
    WARMUP_EPOCHS = cfg.training.get("warmup_epochs", 1)
    USE_COSINE_SCHEDULER = cfg.training.use_cosine_scheduler
    DATASET_NUM_WORKERS = cfg.dataset.get("num_workers", 0)

    # ====== STEP 1: Force load and validate cache metadata at the very beginning ======
    import json
    import re

    project_root = Path(__file__).parent.resolve()
    CACHE_PATH = cfg.dataset.get("dataset_path", None)
    DATASET_CACHE_DIR = cfg.dataset.get("cache_dir", None)

    # Auto-detect cache path if not provided
    if CACHE_PATH is None and DATASET_CACHE_DIR is not None:
        cache_base_dir = Path(DATASET_CACHE_DIR)
        if not cache_base_dir.is_absolute():
            cache_base_dir = project_root / cache_base_dir

        # Get teacher model name from config (will be replaced by metadata later)
        teacher_model_name_config = cfg.teacher.model_name
        safe_model_name = re.sub(r"[^\w\-_]", "_", teacher_model_name_config).strip("_")
        split_config = cfg.dataset.get("split", "val")

        # Try different dataset sizes (tiny, medium, large)
        # For each size, try the configured split first, then fall back to other common splits
        splits_to_try = [split_config] + [s for s in ["train", "val", "test"] if s != split_config]

        for dataset_size_name in ["tiny", "medium", "large"]:
            for split_name in splits_to_try:
                cache_path_candidate = cache_base_dir / safe_model_name / dataset_size_name / split_name
                metadata_file = cache_path_candidate / "metadata.json"
                if metadata_file.exists():
                    CACHE_PATH = str(cache_path_candidate)
                    if is_main_process(rank):
                        if split_name != split_config:
                            print(
                                f"⚠ Warning: Config specifies split '{split_config}' but found cache for "
                                f"'{split_name}'. Using: {CACHE_PATH}"
                            )
                        else:
                            print(f"✓ Auto-detected cache: {CACHE_PATH}")
                    break
            if CACHE_PATH is not None:
                break

    # Assert: Cache path is required
    if CACHE_PATH is None:
        raise ValueError("Cache path is required. Set dataset.cache_path in config.yaml or run prepare_data.py first.")

    # Convert to absolute path
    if not os.path.isabs(CACHE_PATH):
        CACHE_PATH = str(project_root / CACHE_PATH)

    # Assert: Cache metadata file must exist
    metadata_file = Path(CACHE_PATH) / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Cache not found: {CACHE_PATH}. Please run prepare_data.py first to generate cache.")

    # Load cache metadata - MUST exist
    with open(metadata_file, encoding="utf-8") as f:
        cache_metadata = json.load(f)

    # Assert: Cache metadata must contain required fields
    assert cache_metadata is not None, "Cache metadata is required"
    assert "teacher_model_name" in cache_metadata, "Cache metadata must contain teacher_model_name"
    assert "image_embed_dim" in cache_metadata, "Cache metadata must contain image_embed_dim"

    if is_main_process(rank):
        print(f"✓ Cache loaded: {CACHE_PATH}")
        print(f"  Teacher model: {cache_metadata.get('teacher_model_name', 'unknown')}")
        print(f"  Images: {cache_metadata.get('num_images', 0)}")
        print(f"  Captions: {cache_metadata.get('total_captions', 0)}")

    # data_dir is inferred from dataset_path convention: data/coco/cache/... -> data/coco
    # No need to store it in metadata

    LAMBDA_SIGLIP = cfg.loss.lambda_siglip
    LAMBDA_CMD = cfg.loss.lambda_cmd
    LAMBDA_UMD = cfg.loss.lambda_umd
    TEMPERATURE = cfg.loss.temperature

    # Get output directory from Hydra
    from hydra.core.hydra_config import HydraConfig

    output_dir = Path(HydraConfig.get().run.dir)
    checkpoint_path = output_dir / cfg.output.checkpoint_name
    log_every_n_steps = cfg.logging.log_every_n_steps
    eval_every_n_steps = cfg.logging.get("eval_every_n_steps", log_every_n_steps)

    # Early stopping configuration
    early_stopping_enabled = cfg.early_stopping.get("enabled", False)
    early_stopping_patience = cfg.early_stopping.get("patience", 500)
    early_stopping_min_delta = cfg.early_stopping.get("min_delta", 0.001)
    early_stopping_monitor = cfg.early_stopping.get("monitor", "weighted")
    early_stopping_mode = cfg.early_stopping.get("mode", "min")
    restore_best_weights = cfg.early_stopping.get("restore_best_weights", True)

    # Initialize WandB (only on main process)
    wandb_enabled = cfg.wandb.get("enabled", False) and WANDB_AVAILABLE and is_main_process(rank)
    if wandb_enabled:
        wandb_config = cfg.wandb
        run_name = wandb_config.get("name") or f"tinysiglip_{output_dir.name}"
        wandb.init(
            project=wandb_config.get("project", "tinysiglip"),
            entity=wandb_config.get("entity"),
            name=run_name,
            tags=wandb_config.get("tags", []),
            notes=wandb_config.get("notes", ""),
            config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
            dir=str(output_dir),
        )
        print(f"✓ WandB initialized: {wandb.run.url if wandb.run else 'N/A'}")
    elif cfg.wandb.get("enabled", False) and not WANDB_AVAILABLE:
        print("Warning: WandB is enabled in config but not installed. Install with: pip install wandb")

    # Print configuration (only on main process)
    if is_main_process(rank):
        print("=" * 70)
        print("Configuration:")
        print(OmegaConf.to_yaml(cfg))
        print("=" * 70)
        print(f"Output directory: {output_dir}")
        print(f"Checkpoint will be saved to: {checkpoint_path}")
        if use_distributed:
            print(f"Distributed training: {world_size} GPUs")
        print("=" * 70 + "\n")

    # ====== STEP 2: Load processor and get dimensions from cache metadata ======
    # Get teacher model name from cache metadata (required)
    teacher_model_name = cache_metadata["teacher_model_name"]

    # Load processor (single processor, used for both teacher and student)
    if is_main_process(rank):
        print(f"Loading processor from: {teacher_model_name}")
    processor = AutoProcessor.from_pretrained(teacher_model_name)
    if is_main_process(rank):
        print("✓ Processor loaded")

    # Load teacher model to count parameters and get teacher model logit scale and bias
    teacher_total_params = None
    teacher_trainable_params = None
    teacher_vision_params = None
    teacher_text_params = None
    teacher_logit_scale = None
    teacher_logit_bias = None
    if is_main_process(rank):
        print(f"\nLoading teacher model from: {teacher_model_name}")
    try:
        teacher_model = AutoModel.from_pretrained(teacher_model_name)
        teacher_model.eval()  # Set to eval mode
        teacher_total_params = sum(p.numel() for p in teacher_model.parameters())
        teacher_trainable_params = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)

        teacher_logit_scale = teacher_model.logit_scale.exp().item()
        teacher_logit_bias = teacher_model.logit_bias.item()

        print("=" * 30)
        print(f"TEACHER_SCALE = {teacher_logit_scale}")
        print(f"TEACHER_BIAS  = {teacher_logit_bias}")
        print("=" * 30)

        # Count vision and text model parameters separately
        if hasattr(teacher_model, "vision_model"):
            teacher_vision_params = sum(p.numel() for p in teacher_model.vision_model.parameters())
        elif hasattr(teacher_model, "model") and hasattr(teacher_model.model, "vision_model"):
            teacher_vision_params = sum(p.numel() for p in teacher_model.model.vision_model.parameters())
        else:
            # Try to find vision-related parameters by name
            teacher_vision_params = sum(
                p.numel() for name, p in teacher_model.named_parameters() if "vision" in name.lower()
            )

        if hasattr(teacher_model, "text_model"):
            teacher_text_params = sum(p.numel() for p in teacher_model.text_model.parameters())
        elif hasattr(teacher_model, "model") and hasattr(teacher_model.model, "text_model"):
            teacher_text_params = sum(p.numel() for p in teacher_model.model.text_model.parameters())
        else:
            # Try to find text-related parameters by name
            teacher_text_params = sum(
                p.numel() for name, p in teacher_model.named_parameters() if "text" in name.lower()
            )

        if is_main_process(rank):
            print("\n=== Teacher Model Parameters ===")
            print(f"Total parameters: {teacher_total_params:,} ({format_number(teacher_total_params)})")
            print(f"Trainable parameters: {teacher_trainable_params:,} ({format_number(teacher_trainable_params)})")
            if teacher_vision_params is not None:
                print(f"Vision model: {teacher_vision_params:,} ({format_number(teacher_vision_params)})")
            if teacher_text_params is not None:
                print(f"Text model: {teacher_text_params:,} ({format_number(teacher_text_params)})")
            print("================================\n")

        # Free memory - we don't need the teacher model during training (using cached embeddings)
        del teacher_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        if is_main_process(rank):
            print(f"Warning: Could not load teacher model to count parameters: {e}")

    # Get dimensions from cache metadata (required)
    teacher_projection_dim = cache_metadata["image_embed_dim"]
    # Get vocab size from processor tokenizer
    vocab_size = len(processor.tokenizer) if hasattr(processor.tokenizer, "__len__") else 32000

    if is_main_process(rank):
        print("✓ Using dimensions from cache metadata:")
        print(f"  Projection dim: {teacher_projection_dim}")
        print(f"  Vocab size: {vocab_size}")

    # According to plan.md: Student output dimension must match Teacher's projection_dim
    # Use projection_dim from cache metadata (no need to read from config)
    STUDENT_PROJECTION_DIM = cast(int, teacher_projection_dim)
    STUDENT_VOCAB_SIZE = cast(int, vocab_size)

    if is_main_process(rank):
        print(f"\n✓ Student projection_dim: {STUDENT_PROJECTION_DIM}")
        print(f"✓ Student vocab_size: {STUDENT_VOCAB_SIZE} (from cache metadata)")

    # Create student model config
    if is_main_process(rank):
        print("Creating student model config...")
    student_config = TinySiglipConfig(
        vision_model_name=cfg.student.vision_model_name,
        text_vocab_size=STUDENT_VOCAB_SIZE,
        text_seq_len=TEXT_SEQ_LEN,
        text_dim=cfg.student.text_dim,
        text_layers=cfg.student.text_layers,
        text_nhead=cfg.student.text_nhead,
        text_ff_dim_multiplier=cfg.student.text_ff_dim_multiplier,
        projection_dim=STUDENT_PROJECTION_DIM,  # Use teacher's projection_dim
    )

    # Create student model
    if is_main_process(rank):
        print("Creating student model...")
    student_model = TinySiglipModel(config=student_config).to(device)

    # Count parameters
    vision_backbone_params = count_parameters(student_model.vision_model.encoder)
    vision_proj_params = count_parameters(student_model.vision_model.projection)
    vision_params = vision_backbone_params + vision_proj_params

    # Text model parameters (including position embedding which is a Parameter)
    text_embedding_params = count_parameters(student_model.text_model.embedding)
    text_transformer_params = count_parameters(student_model.text_model.transformer)
    text_proj_params = count_parameters(student_model.text_model.projection)
    text_pos_params = student_model.text_model.pos_embedding.numel()  # Position embedding is a Parameter
    text_params = text_embedding_params + text_transformer_params + text_proj_params + text_pos_params

    total_params = count_parameters(student_model)

    if is_main_process(rank):
        print("\n=== Model Parameters ===")
        print("Vision model parameters:")
        print(f"  - Backbone: {vision_backbone_params:,} ({format_number(vision_backbone_params)})")
        print(f"  - Projection: {vision_proj_params:,} ({format_number(vision_proj_params)})")
        print(f"  - Total: {vision_params:,} ({format_number(vision_params)})")
        print("\nText model parameters:")
        print(f"  - Embedding: {text_embedding_params:,} ({format_number(text_embedding_params)})")
        print(f"  - Position embedding: {text_pos_params:,} ({format_number(text_pos_params)})")
        print(f"  - Transformer: {text_transformer_params:,} ({format_number(text_transformer_params)})")
        print(f"  - Projection: {text_proj_params:,} ({format_number(text_proj_params)})")
        print(f"  - Total: {text_params:,} ({format_number(text_params)})")
        print(f"\nTotal student model parameters: {total_params:,} ({format_number(total_params)})")
        print("========================\n")

        # Compare teacher and student model parameters
        if teacher_total_params is not None:
            print("=== Teacher vs Student Model Comparison ===")
            print(f"Teacher model:  {teacher_total_params:,} ({format_number(teacher_total_params)})")
            if teacher_vision_params is not None:
                print(f"  - Vision: {teacher_vision_params:,} ({format_number(teacher_vision_params)})")
            if teacher_text_params is not None:
                print(f"  - Text: {teacher_text_params:,} ({format_number(teacher_text_params)})")
            print(f"Student model:  {total_params:,} ({format_number(total_params)})")
            print(f"  - Vision: {vision_params:,} ({format_number(vision_params)})")
            print(f"  - Text: {text_params:,} ({format_number(text_params)})")
            compression_ratio = teacher_total_params / total_params if total_params > 0 else 0
            size_reduction = (1 - total_params / teacher_total_params) * 100 if teacher_total_params > 0 else 0
            print(f"Compression ratio: {compression_ratio:.2f}x")
            print(f"Size reduction:   {size_reduction:.2f}%")
            print("==========================================\n")

        # Log model parameters to WandB
        if wandb_enabled:
            # Teacher model parameters
            if teacher_total_params is not None:
                wandb.summary["teacher/total_params"] = teacher_total_params
                wandb.summary["teacher/total_params_M"] = teacher_total_params / 1e6
                if teacher_vision_params is not None:
                    wandb.summary["teacher/vision_params"] = teacher_vision_params
                    wandb.summary["teacher/vision_params_M"] = teacher_vision_params / 1e6
                if teacher_text_params is not None:
                    wandb.summary["teacher/text_params"] = teacher_text_params
                    wandb.summary["teacher/text_params_M"] = teacher_text_params / 1e6

            # Student model parameters
            wandb.summary["student/total_params"] = total_params
            wandb.summary["student/total_params_M"] = total_params / 1e6
            wandb.summary["student/vision_params"] = vision_params
            wandb.summary["student/vision_params_M"] = vision_params / 1e6
            wandb.summary["student/text_params"] = text_params
            wandb.summary["student/text_params_M"] = text_params / 1e6
            wandb.summary["student/vision_backbone_params"] = vision_backbone_params
            wandb.summary["student/vision_proj_params"] = vision_proj_params
            wandb.summary["student/text_embedding_params"] = text_embedding_params
            wandb.summary["student/text_transformer_params"] = text_transformer_params
            wandb.summary["student/text_proj_params"] = text_proj_params
            wandb.summary["student/text_pos_params"] = text_pos_params

            # Compression metrics
            if teacher_total_params is not None:
                compression_ratio = teacher_total_params / total_params if total_params > 0 else 0
                size_reduction = (1 - total_params / teacher_total_params) * 100 if teacher_total_params > 0 else 0
                wandb.summary["compression/ratio"] = compression_ratio
                wandb.summary["compression/size_reduction_pct"] = size_reduction

    # Wrap model with DDP if using distributed training
    if use_distributed:
        student_model = DDP(
            student_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )
        if is_main_process(rank):
            print("✓ Student model wrapped with DistributedDataParallel")

    # Use fixed logit scale (temperature) - no need to learn it in distillation
    # Original CLIP uses learnable logit_scale, but for distillation we can use a fixed value
    logit_scale = torch.tensor(math.log(1 / 0.07), device=device)  # Fixed value, not a parameter

    # Optimizer: use unwrapped model parameters (DDP wraps them, making them non-leaf)
    # Get the underlying module if wrapped with DDP, otherwise use the model directly
    model_for_optimizer = student_model.module if isinstance(student_model, DDP) else student_model

    # Collect model parameters and verify they are all leaf tensors
    # This is critical: optimizer can only optimize leaf tensors
    model_params = []
    non_leaf_count = 0
    for name, param in model_for_optimizer.named_parameters():
        if param.requires_grad:
            if param.is_leaf:
                model_params.append(param)
            else:
                non_leaf_count += 1
                if is_main_process(rank) and non_leaf_count <= 5:
                    print(f"Warning: Skipping non-leaf parameter: {name}, shape={param.shape}")

    if non_leaf_count > 0 and is_main_process(rank):
        print(f"Warning: Skipped {non_leaf_count} non-leaf parameters that require gradients")

    if is_main_process(rank):
        print(
            f"Optimizer will optimize {len(model_params)} model parameters "
            f"(using fixed logit_scale={logit_scale.item():.4f})"
        )

    optimizer = torch.optim.AdamW(model_params, lr=LEARNING_RATE)

    # Calculate steps per epoch and total steps
    # Note: We'll calculate this after dataloader is created, so we'll initialize scheduler later
    steps_per_epoch = None
    total_steps = None
    warmup_steps = None

    # ====== STEP 3: Setup dataset (always use real data from cache) ======
    if is_main_process(rank):
        print("\n=== Setting up COCO Dataset ===")

    # Use single processor for both student and teacher
    dataset = COCOCaptionDataset(
        dataset_path=CACHE_PATH,  # Required: path to cached embeddings
        processor=processor,  # Single processor for both student and teacher
        max_seq_len=TEXT_SEQ_LEN,
        verbose=is_main_process(rank),
    )

    # Use DistributedSampler for distributed training
    sampler = None
    if use_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False if use_distributed else True,
        sampler=sampler,
        collate_fn=collate_coco_batch,
        num_workers=DATASET_NUM_WORKERS,
        pin_memory=True if (torch.cuda.is_available() and device.type == "cuda") else False,
    )

    # Print dataset info
    if is_main_process(rank):
        print(f"✓ COCO dataset loaded: {len(dataset)} samples")
        print(f"  Steps per epoch: {len(dataloader)} (batch_size={BATCH_SIZE}, world_size={world_size})")
        print(f"✓ Using cached embeddings from: {CACHE_PATH}")
        print("=" * 30 + "\n")

    # Calculate steps per epoch and total steps for scheduler
    steps_per_epoch = len(dataloader)
    total_steps = NUM_EPOCHS * steps_per_epoch
    warmup_steps = int(WARMUP_EPOCHS * steps_per_epoch)

    if is_main_process(rank):
        print("\n=== Training Configuration ===")
        print(f"Dataset size: {len(dataset)} samples")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"World size: {world_size}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Number of epochs: {NUM_EPOCHS}")
        print(f"Total training steps: {total_steps}")
        print(f"Warmup epochs: {WARMUP_EPOCHS}")
        print(f"Warmup steps: {warmup_steps}")
        print("=" * 30 + "\n")

    # Learning rate scheduler (HuggingFace style with warmup)
    if USE_COSINE_SCHEDULER:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        if is_main_process(rank):
            print("✓ Cosine learning rate scheduler with warmup initialized (HuggingFace style)")
            print(f"  Initial LR: {LEARNING_RATE}")
            print(f"  Warmup steps: {warmup_steps} ({WARMUP_EPOCHS} epochs)")
            print(f"  Total steps: {total_steps} ({NUM_EPOCHS} epochs)")
            print("  Final LR will decay to: ~0 (cosine decay)")
    else:
        scheduler = None

    # Training loop with epochs
    # Get the underlying model from DDP wrapper if using distributed training
    if use_distributed:
        model_to_train = cast(nn.Module, student_model.module)
    else:
        model_to_train = student_model
    model_to_train.train()

    # Compute FLOPs (only once, on main process, after model is ready)
    flops_metrics = {}
    if is_main_process(rank):
        print("\n=== Computing Model FLOPs ===")
        flops_metrics = compute_flops(model_to_train, IMAGE_SIZE, TEXT_SEQ_LEN, device)
        if flops_metrics.get("total_flops") is not None:
            print(f"Total FLOPs: {flops_metrics['total_flops']:.2e}")
            print(f"Total GFLOPs: {flops_metrics['total_flops_g']:.2f}")
            if "note" in flops_metrics:
                print(f"Note: {flops_metrics['note']}")
        else:
            print("Could not compute FLOPs")
        print("=" * 30 + "\n")

    if is_main_process(rank):
        print(f"\nStarting training for {NUM_EPOCHS} epochs...")
        print(f"Total steps: {total_steps} ({steps_per_epoch} steps per epoch)")

    # Initialize performance tracking
    performance_tracker = {
        "samples_seen": 0,
        "tokens_seen": 0,
        "epoch_start_time": None,
        "step_times": [],
        "batch_times": [],
    }

    # Initialize early stopping
    best_monitor_value = None
    best_step = 0
    best_model_state = None

    if early_stopping_enabled and is_main_process(rank):
        print("\n=== Early Stopping Enabled ===")
        print(f"Monitor: {early_stopping_monitor}")
        print(f"Mode: {early_stopping_mode}")
        print(f"Patience: {early_stopping_patience} steps")
        print(f"Min delta: {early_stopping_min_delta}")
        print(f"Restore best weights: {restore_best_weights}")
        print("=" * 30 + "\n")

    # Training loop over epochs
    global_step = 0
    should_stop = False

    # Progress bar for epochs (only on main process)
    pbar = tqdm(total=NUM_EPOCHS, desc="Training", disable=not is_main_process(rank))

    for epoch in range(NUM_EPOCHS):
        # Set epoch for DistributedSampler to ensure proper shuffling
        if use_distributed and sampler is not None:
            sampler.set_epoch(epoch)

        # Track epoch start time
        epoch_start_time = time.time()
        performance_tracker["epoch_start_time"] = epoch_start_time

        # Epoch-level progress bar
        epoch_pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
            disable=not is_main_process(rank),
            leave=False,
        )

        epoch_loss_sum = 0.0
        epoch_loss_dict_sum = {}

        for batch_idx, batch in enumerate(epoch_pbar):
            # Track batch processing time
            batch_start_time = time.time()

            # COCO dataset returns a dict
            student_images = batch["student_images"].to(device)
            student_text_ids = batch["student_text_ids"].to(device)
            student_attention_mask = batch["student_attention_mask"].to(device)
            # Note: teacher_images and teacher_text_ids may not be in batch if using cache

            # Track samples and tokens
            batch_size = student_images.size(0)
            actual_batch_size = batch_size * world_size if use_distributed else batch_size
            performance_tracker["samples_seen"] += actual_batch_size
            # Count non-padding tokens (assuming 0 is padding token)
            num_tokens_batch = (student_text_ids > 0).sum().item()
            num_tokens = num_tokens_batch * world_size if use_distributed else num_tokens_batch
            performance_tracker["tokens_seen"] += num_tokens

            optimizer.zero_grad()

            # Get teacher embeddings from cache (required - cache is mandatory)
            # Cache is always used, no need to check or compute on-the-fly
            teacher_image_features = batch["teacher_image_embeds"].to(device)
            teacher_text_features = batch["teacher_text_embeds"].to(device)

            # Student forward (uses student vocab token IDs)
            student_image_features, student_text_features, student_logit_scale, student_logit_bias = model_to_train(
                student_images, student_text_ids, student_attention_mask
            )

            # Compute loss (simplified: directly compare projected features)
            loss, loss_dict = compute_total_loss(
                student_image_features=student_image_features,
                student_text_features=student_text_features,
                teacher_image_features=teacher_image_features,
                teacher_text_features=teacher_text_features,
                student_logit_scale=student_logit_scale,
                student_logit_bias=student_logit_bias,
                teacher_logit_scale=cast(float, teacher_logit_scale),
                teacher_logit_bias=cast(float, teacher_logit_bias),
                lambda_siglip=LAMBDA_SIGLIP,
                lambda_cmd=LAMBDA_CMD,
                lambda_umd=LAMBDA_UMD,
                temperature=TEMPERATURE,
            )

            # Backward
            backward_start_time = time.time()
            loss.backward()
            optimizer.step()

            # Update learning rate
            if scheduler is not None:
                scheduler.step()

            # Track timing
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            step_time = batch_end_time - backward_start_time  # Time for backward + optimizer step
            performance_tracker["batch_times"].append(batch_time)
            performance_tracker["step_times"].append(step_time)
            # Keep only last 100 timings for rolling average
            if len(performance_tracker["batch_times"]) > 100:
                performance_tracker["batch_times"] = performance_tracker["batch_times"][-100:]
                performance_tracker["step_times"] = performance_tracker["step_times"][-100:]

            global_step += 1

            # Accumulate epoch losses
            epoch_loss_sum += loss_dict["weighted"]
            for key, value in loss_dict.items():
                if key not in epoch_loss_dict_sum:
                    epoch_loss_dict_sum[key] = 0.0
                epoch_loss_dict_sum[key] += value

            # Compute evaluation metrics periodically
            eval_metrics = {}
            if global_step % eval_every_n_steps == 0:
                with torch.no_grad():
                    eval_metrics = compute_evaluation_metrics(
                        student_image_features=student_image_features,
                        student_text_features=student_text_features,
                        teacher_image_features=teacher_image_features,
                        teacher_text_features=teacher_text_features,
                        student_logit_scale=student_logit_scale,
                        student_logit_bias=student_logit_bias,
                        teacher_logit_scale=cast(float, teacher_logit_scale),
                        teacher_logit_bias=cast(float, teacher_logit_bias),
                    )

            # Early stopping check (check every eval_every_n_steps)
            if early_stopping_enabled and global_step % eval_every_n_steps == 0:
                # Get the monitored value from loss_dict or eval_metrics
                if early_stopping_monitor in loss_dict:
                    monitor_value = loss_dict[early_stopping_monitor]
                elif early_stopping_monitor in eval_metrics:
                    monitor_value = eval_metrics[early_stopping_monitor]
                else:
                    # Default to weighted loss if monitor key not found
                    monitor_value = loss_dict.get("weighted", float("inf"))
                    if is_main_process(rank) and global_step == eval_every_n_steps:
                        print(
                            f"Warning: Monitor key '{early_stopping_monitor}' not found, using 'weighted' loss instead"
                        )

                # Check if this is a better value
                is_better = False
                if best_monitor_value is None:
                    is_better = True
                elif early_stopping_mode == "min":
                    is_better = monitor_value < (best_monitor_value - early_stopping_min_delta)
                else:  # mode == "max"
                    is_better = monitor_value > (best_monitor_value + early_stopping_min_delta)

                if is_better:
                    best_monitor_value = monitor_value
                    best_step = global_step
                    # Save best model state (only on main process)
                    if is_main_process(rank):
                        best_model_state = model_to_train.state_dict().copy()
                        if global_step % log_every_n_steps == 0:
                            print(
                                f"✓ New best {early_stopping_monitor}: {monitor_value:.6f} "
                                f"(step {global_step}, epoch {epoch + 1})"
                            )
                else:
                    # Calculate steps since last improvement
                    steps_since_best = global_step - best_step
                    if steps_since_best >= early_stopping_patience:
                        should_stop = True
                        if is_main_process(rank):
                            print(f"\n{'=' * 70}")
                            print("Early stopping triggered!")
                            print(f"  Best {early_stopping_monitor}: {best_monitor_value:.6f} at step {best_step}")
                            print(f"  Current {early_stopping_monitor}: {monitor_value:.6f} at step {global_step}")
                            print(
                                f"  No improvement for {steps_since_best} steps (patience: {early_stopping_patience})"
                            )
                            print(f"{'=' * 70}\n")

            # Compute performance metrics
            recent_batch_times = performance_tracker["batch_times"][-10:]
            avg_batch_time = sum(recent_batch_times) / min(10, len(recent_batch_times))
            samples_per_sec = actual_batch_size / avg_batch_time if avg_batch_time > 0 else 0
            tokens_per_sec = num_tokens / avg_batch_time if avg_batch_time > 0 else 0

            # Update epoch progress bar
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_pbar.set_postfix(
                {
                    **{k: f"{v / (batch_idx + 1):.4f}" for k, v in epoch_loss_dict_sum.items()},
                    "lr": f"{current_lr:.2e}",
                    "samples/s": f"{samples_per_sec:.1f}",
                    "tokens/s": f"{tokens_per_sec:.0f}",
                }
            )

            # Compute performance metrics for logging
            recent_batch_times = performance_tracker["batch_times"][-10:]
            recent_step_times = performance_tracker["step_times"][-10:]
            avg_batch_time = sum(recent_batch_times) / min(10, len(recent_batch_times))
            avg_step_time = sum(recent_step_times) / min(10, len(recent_step_times))
            samples_per_sec = actual_batch_size / avg_batch_time if avg_batch_time > 0 else 0
            tokens_per_sec = num_tokens / avg_batch_time if avg_batch_time > 0 else 0
            images_per_sec = actual_batch_size / avg_batch_time if avg_batch_time > 0 else 0

            # Log to WandB
            if wandb_enabled:
                wandb_log = {
                    "train/loss": loss_dict["weighted"],
                    "train/loss_siglip": loss_dict["siglip"],
                    "train/loss_cmd": loss_dict["cmd"],
                    "train/loss_umd": loss_dict["umd"],
                    "train/learning_rate": current_lr,
                    "train/step": global_step,
                    "train/epoch": epoch + 1,
                    "train/samples_seen": performance_tracker["samples_seen"],
                    "train/tokens_seen": performance_tracker["tokens_seen"],
                    "train/samples_per_sec": samples_per_sec,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/images_per_sec": images_per_sec,
                    "train/batch_time_ms": avg_batch_time * 1000,
                    "train/step_time_ms": avg_step_time * 1000,
                }
                # Add FLOPs metrics if available (log once per epoch)
                if batch_idx == 0 and flops_metrics.get("total_flops_g") is not None:
                    wandb_log["model/flops_g"] = flops_metrics["total_flops_g"]
                # Add evaluation metrics if available
                if eval_metrics:
                    wandb_log.update({f"eval/{k}": v for k, v in eval_metrics.items()})
                wandb.log(wandb_log, step=global_step)

            # Logging every N steps
            if global_step % log_every_n_steps == 0 and is_main_process(rank):
                log_message = (
                    f"\nEpoch {epoch + 1}/{NUM_EPOCHS}, Step {global_step}/{total_steps}: "
                    f"Loss: {loss_dict} | LR: {current_lr:.2e}"
                )
                # Add performance metrics
                log_message += (
                    f"\n  Performance: {samples_per_sec:.1f} samples/s, "
                    f"{tokens_per_sec:.0f} tokens/s, {images_per_sec:.1f} images/s | "
                    f"Batch time: {avg_batch_time * 1000:.1f}ms | "
                    f"Samples seen: {performance_tracker['samples_seen']:,}"
                )
                if eval_metrics:
                    # Format metrics nicely
                    retrieval_metrics = {k: f"{v:.3f}" for k, v in eval_metrics.items() if "recall@" in k}
                    similarity_metrics = {
                        k: f"{v:.3f}" for k, v in eval_metrics.items() if "sim" in k or "correlation" in k
                    }
                    if retrieval_metrics:
                        log_message += f"\n  Retrieval: {retrieval_metrics}"
                    if similarity_metrics:
                        log_message += f"\n  Similarity: {similarity_metrics}"
                if early_stopping_enabled:
                    steps_since_best = global_step - best_step if best_step > 0 else 0
                    log_message += (
                        f" | Best {early_stopping_monitor}: {best_monitor_value:.6f} (step {best_step}) "
                        f"| Steps since best: {steps_since_best}/{early_stopping_patience}"
                    )
                print(log_message)

            # Check early stopping
            if should_stop:
                break

        # Close epoch progress bar
        epoch_pbar.close()

        # Calculate epoch average losses
        epoch_avg_loss_dict = {k: v / len(dataloader) for k, v in epoch_loss_dict_sum.items()}

        # Calculate epoch performance metrics
        epoch_time = time.time() - epoch_start_time
        epoch_samples_per_sec = len(dataset) / epoch_time if epoch_time > 0 else 0
        batch_times = performance_tracker["batch_times"]
        epoch_avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0

        # Update main progress bar
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            {
                **{k: f"{v:.4f}" for k, v in epoch_avg_loss_dict.items()},
                "best": f"{best_monitor_value:.4f}" if best_monitor_value is not None else "N/A",
                "lr": f"{current_lr:.2e}",
                "samples/s": f"{epoch_samples_per_sec:.1f}",
            }
        )
        pbar.update(1)

        # Log epoch summary
        if is_main_process(rank):
            log_message = (
                f"\nEpoch {epoch + 1}/{NUM_EPOCHS} completed: Avg Loss: {epoch_avg_loss_dict} | LR: {current_lr:.2e}"
            )
            log_message += (
                f"\n  Epoch time: {epoch_time:.1f}s | "
                f"Throughput: {epoch_samples_per_sec:.1f} samples/s | "
                f"Avg batch time: {epoch_avg_batch_time * 1000:.1f}ms | "
                f"Samples seen: {performance_tracker['samples_seen']:,} | "
                f"Tokens: {performance_tracker['tokens_seen']:,}"
            )
            if early_stopping_enabled:
                steps_since_best = global_step - best_step if best_step > 0 else 0
                log_message += (
                    f" | Best {early_stopping_monitor}: {best_monitor_value:.6f} (step {best_step}) "
                    f"| Steps since best: {steps_since_best}/{early_stopping_patience}"
                )
            print(log_message)

        # Check early stopping
        if should_stop:
            break

    pbar.close()

    # Restore best weights if early stopping was enabled and triggered
    if early_stopping_enabled and should_stop and restore_best_weights and is_main_process(rank):
        if best_model_state is not None:
            print(f"\nRestoring best model weights from step {best_step}...")
            model_to_train.load_state_dict(best_model_state)
            print(f"✓ Best weights restored (best {early_stopping_monitor}: {best_monitor_value:.6f})")

    if is_main_process(rank):
        if should_stop:
            best_epoch = best_step // steps_per_epoch if steps_per_epoch > 0 else 0
            print(
                f"\nTraining stopped early after {global_step} steps "
                f"(best at step {best_step}, epoch {best_epoch + 1})!"
            )
        else:
            print(f"\nTraining completed after {NUM_EPOCHS} epochs ({global_step} total steps)!")

    # Log final metrics to WandB
    if wandb_enabled:
        epoch_start = performance_tracker.get("epoch_start_time")
        total_training_time = time.time() - epoch_start if epoch_start is not None else time.time()
        wandb.summary["final_step"] = global_step
        wandb.summary["final_epoch"] = epoch + 1
        wandb.summary["total_steps"] = total_steps
        wandb.summary["total_epochs"] = NUM_EPOCHS
        wandb.summary["total_samples_seen"] = performance_tracker["samples_seen"]
        wandb.summary["total_tokens_seen"] = performance_tracker["tokens_seen"]
        wandb.summary["total_training_time_seconds"] = total_training_time
        if flops_metrics.get("total_flops_g") is not None:
            flops_g = flops_metrics["total_flops_g"]
            if isinstance(flops_g, (int, float)):
                wandb.summary["model/flops_g"] = flops_g
            flops_total = flops_metrics["total_flops"]
            if isinstance(flops_total, (int, float)):
                wandb.summary["model/flops"] = flops_total

    # Save checkpoint only on main process
    if is_main_process(rank):
        print(f"Saving model to {checkpoint_path}...")
        # Get state dict from DDP model if needed
        # Use best model state if early stopping was enabled and best weights were restored
        if early_stopping_enabled and restore_best_weights and best_model_state is not None:
            model_state_dict = best_model_state
            # logit_scale is fixed, so use the current value
            logit_scale_data = logit_scale.item() if logit_scale.numel() == 1 else logit_scale
            best_epoch = best_step // steps_per_epoch if steps_per_epoch > 0 else 0
            print(
                f"  Saving best model from step {best_step} (epoch {best_epoch + 1}) "
                f"(best {early_stopping_monitor}: {best_monitor_value:.6f})"
            )
        else:
            model_state_dict = model_to_train.state_dict()
            # logit_scale is now a fixed tensor, not a Parameter
            logit_scale_data = logit_scale.item() if logit_scale.numel() == 1 else logit_scale

        checkpoint_dict = {
            "student_model": model_state_dict,
            "logit_scale": logit_scale_data,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "model_config": student_config.to_dict(),  # Save model architecture config
        }
        if early_stopping_enabled:
            checkpoint_dict["best_step"] = best_step
            checkpoint_dict["best_epoch"] = best_step // steps_per_epoch if steps_per_epoch > 0 else 0
            checkpoint_dict["best_monitor_value"] = best_monitor_value
            checkpoint_dict["best_monitor_key"] = early_stopping_monitor

        torch.save(checkpoint_dict, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

        # Save configuration to output directory
        config_save_path = output_dir / "config.yaml"
        OmegaConf.save(cfg, config_save_path)
        print(f"Configuration saved to {config_save_path}")

        # Save model architecture config as JSON
        model_config_path = output_dir / "model_config.json"
        student_config.save_json(model_config_path)
        print(f"Model architecture config saved to {model_config_path}")

        # Save student processor for inference (Hugging Face style)
        print("\n=== Saving Student Processor for Inference ===")

        # Reset augmentation to False for inference
        if hasattr(processor.image_processor, "use_augmentation"):
            processor.image_processor.use_augmentation = False
            processor.image_processor._build_transform()

        processor_save_path = output_dir / "processor"
        try:
            processor.save_pretrained(str(processor_save_path))
            print(f"✓ Processor saved to {processor_save_path}")
            print(f"  - Tokenizer: {processor_save_path / 'tokenizer'}")
            print(f"  - Image processor: {processor_save_path / 'image_processor'}")
        except Exception as e:
            print(f"Warning: Could not save processor: {e}")

        print("=" * 50)

        # Finish WandB run
        if wandb_enabled:
            wandb.finish()
            print("✓ WandB run completed")

    # Cleanup distributed training
    cleanup_distributed()


if __name__ == "__main__":
    main()  # Hydra decorator handles configuration loading
