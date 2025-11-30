"""
Simplified training script for SigLIP distillation with Hydra configuration.
"""

import math
import os
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
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    SiglipModel,
    get_cosine_schedule_with_warmup,
)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

from tinysiglip.coco_dataset import COCOCaptionDataset, collate_coco_batch
from tinysiglip.embedding_distillation import (
    create_dummy_token_mapping,
    create_token_mapping,
    transfer_embedding_weights,
)
from tinysiglip.fake_dataset import DummySiglipDataset
from tinysiglip.loss import compute_total_loss
from tinysiglip.metrics import compute_evaluation_metrics
from tinysiglip.model import TinySiglipModel
from tinysiglip.processor import TinySiglipProcessor


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


def get_teacher_model(model_name="google/siglip2-base-patch16-224", device=None) -> SiglipModel:
    """Load teacher SigLIP model."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # macOS Metal Performance Shaders (Apple Silicon)
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    try:
        from transformers import SiglipModel

        model = SiglipModel.from_pretrained(model_name).to(device)  # pyright: ignore[reportArgumentType]
    except ImportError:
        # Fallback to AutoModel if SiglipModel is not available
        model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return cast(SiglipModel, model)


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
    TEACHER_MODEL_NAME = cfg.teacher.model_name
    BATCH_SIZE = cfg.training.batch_size
    IMAGE_SIZE = cfg.training.image_size
    TEXT_SEQ_LEN = cfg.training.text_seq_len
    MAX_STEPS = cfg.training.max_steps
    LEARNING_RATE = cfg.training.learning_rate
    WARMUP_STEPS = cfg.training.warmup_steps
    USE_COSINE_SCHEDULER = cfg.training.use_cosine_scheduler

    USE_REAL_DATA = cfg.dataset.use_real_data
    DATASET_SPLIT = cfg.dataset.split
    DATASET_CACHE_DIR = cfg.dataset.cache_dir
    USE_AUGMENTATION = cfg.dataset.use_augmentation
    DATASET_NUM_WORKERS = cfg.dataset.get("num_workers", 0)

    STUDENT_VOCAB_SIZE = cfg.student.vocab_size
    STUDENT_TOKENIZER_NAME = cfg.student.tokenizer_name

    LAMBDA_CMD = cfg.loss.lambda_cmd
    LAMBDA_UMD = cfg.loss.lambda_umd
    TEMPERATURE = cfg.loss.temperature
    USE_WEIGHT_TRANSFER = cfg.embedding.use_weight_transfer

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

    # Load teacher model and processor
    if is_main_process(rank):
        print("Loading teacher model...")
    teacher_model = get_teacher_model(TEACHER_MODEL_NAME, device=device)
    teacher_config = teacher_model.config

    # Load teacher processor (recommended for proper preprocessing)
    if is_main_process(rank):
        print("Loading teacher processor...")
    try:
        teacher_processor = AutoProcessor.from_pretrained(TEACHER_MODEL_NAME)
        if is_main_process(rank):
            print("✓ Teacher processor loaded")
    except Exception as e:
        if is_main_process(rank):
            print(f"Warning: Could not load teacher processor: {e}")
        teacher_processor = None

    # Load tokenizers and create student processor for real data
    student_tokenizer = None
    teacher_tokenizer = None
    student_processor = None
    if USE_REAL_DATA:
        if is_main_process(rank):
            print("Loading tokenizers for real data...")

        # Load teacher tokenizer (from processor or separately)
        if teacher_processor is not None:
            try:
                teacher_tokenizer = teacher_processor.tokenizer
                print("✓ Teacher tokenizer loaded from processor")
            except Exception as e:
                print(f"Warning: Could not get tokenizer from processor: {e}")
                teacher_tokenizer = None

        if teacher_tokenizer is None:
            try:
                teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
                print("✓ Teacher tokenizer loaded directly")
            except Exception as e:
                error_msg = str(e)
                print(f"Error: Could not load teacher tokenizer: {error_msg}")
                if "protobuf" in error_msg.lower() or "sentencepiece" in error_msg.lower():
                    print("\n" + "=" * 70)
                    print("MISSING DEPENDENCY DETECTED")
                    print("=" * 70)
                    print("To fix this issue, please install the missing dependencies:")
                    print("  uv add protobuf sentencepiece")
                    print("  OR")
                    print("  pip install protobuf sentencepiece")
                    print("\nAlternatively, you can:")
                    print("  1. Set USE_REAL_DATA = False to use dummy data")
                    print("  2. Or install dependencies and restart the script")
                    print("=" * 70 + "\n")
                teacher_tokenizer = None

        # Load student tokenizer
        if STUDENT_TOKENIZER_NAME is not None:
            try:
                student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_TOKENIZER_NAME)
                print(f"✓ Student tokenizer loaded: {STUDENT_TOKENIZER_NAME}")
            except Exception as e:
                print(f"Warning: Could not load student tokenizer {STUDENT_TOKENIZER_NAME}: {e}")
                student_tokenizer = teacher_tokenizer  # Fallback to teacher tokenizer
        else:
            # Use teacher tokenizer as fallback
            student_tokenizer = teacher_tokenizer
            if student_tokenizer is not None:
                print("Using teacher tokenizer for student (no separate student tokenizer specified)")

        if student_tokenizer is None:
            print("\n" + "=" * 70)
            print("ERROR: Could not load any tokenizer for real data")
            print("=" * 70)
            print("Options:")
            print("  1. Install missing dependencies:")
            print("     uv add protobuf sentencepiece")
            print("  2. Use dummy data instead:")
            print("     Set USE_REAL_DATA = False in train.py")
            print("=" * 70 + "\n")
            raise ValueError("Could not load any tokenizer. Cannot use real data without tokenizer.")

        # Create student processor from tokenizer
        if student_tokenizer is not None:
            # Extract image processor from teacher processor or create new one
            student_image_processor = None
            if teacher_processor is not None:
                if hasattr(teacher_processor, "image_processor"):
                    student_image_processor = teacher_processor.image_processor
                elif hasattr(teacher_processor, "feature_extractor"):
                    student_image_processor = teacher_processor.feature_extractor

            # Create student processor
            student_processor = TinySiglipProcessor(
                tokenizer=student_tokenizer,
                image_processor=student_image_processor,
                image_size=IMAGE_SIZE,
                max_seq_len=TEXT_SEQ_LEN,
                use_augmentation=False,  # Will be set based on split when creating dataset
            )
            print("✓ Student processor created")

    # Get teacher dimensions
    teacher_projection_dim = getattr(teacher_config, "projection_dim", 768)

    # Try different ways to access vision/text dimensions
    if hasattr(teacher_config, "vision_config"):
        teacher_vision_dim = teacher_config.vision_config.hidden_size
    elif hasattr(teacher_config, "vision_hidden_size"):
        teacher_vision_dim = teacher_config.vision_hidden_size
    else:
        teacher_vision_dim = 768  # Default fallback

    if hasattr(teacher_config, "text_config"):
        teacher_text_dim = teacher_config.text_config.hidden_size
        teacher_vocab_size = teacher_config.text_config.vocab_size
    elif hasattr(teacher_config, "text_hidden_size"):
        teacher_text_dim = teacher_config.text_hidden_size
        teacher_vocab_size = getattr(teacher_config, "vocab_size", 32000)
    else:
        teacher_text_dim = 768  # Default fallback
        teacher_vocab_size = 32000  # Default fallback

    # Get actual vocab size from model if available
    if hasattr(teacher_model, "text_model") and hasattr(teacher_model.text_model, "embeddings"):
        embeddings = teacher_model.text_model.embeddings
        if hasattr(embeddings, "token_embedding"):
            embed_layer = embeddings.token_embedding
            if isinstance(embed_layer, nn.Module) and hasattr(embed_layer, "num_embeddings"):
                teacher_vocab_size = cast(int, embed_layer.num_embeddings)
        elif hasattr(embeddings, "word_embeddings"):
            embed_layer = embeddings.word_embeddings
            if isinstance(embed_layer, nn.Module) and hasattr(embed_layer, "num_embeddings"):
                teacher_vocab_size = cast(int, embed_layer.num_embeddings)

    if is_main_process(rank):
        print(f"Teacher projection dim: {teacher_projection_dim}")
        print(f"Teacher vision dim: {teacher_vision_dim}")
        print(f"Teacher text dim: {teacher_text_dim}")
        print(f"Teacher vocab size: {teacher_vocab_size}")

    # Determine student vocab size (use configured value or teacher's vocab size)
    if STUDENT_VOCAB_SIZE is None:
        STUDENT_VOCAB_SIZE = cast(int, teacher_vocab_size)
        if is_main_process(rank):
            print(f"\nStudent vocab size: {STUDENT_VOCAB_SIZE} (same as teacher)")
    else:
        if is_main_process(rank):
            print(f"\nStudent vocab size: {STUDENT_VOCAB_SIZE} (English-only)")
            print("Note: In real training, use tokenizers to map text to token IDs for each model")

    # Calculate valid token ID range (for dummy data only)
    # Use the smaller of the two vocab sizes for token generation
    # In real training, you'd use a tokenizer to map between vocabularies
    max_valid_token_id_student = max(1, STUDENT_VOCAB_SIZE - 10)
    max_valid_token_id_teacher = max(1, cast(int, teacher_vocab_size) - 10)

    # Create student model
    if is_main_process(rank):
        print("Creating student model...")
    student_model = TinySiglipModel(
        vision_model_name=cfg.student.vision_model_name,
        vision_dim=cfg.student.vision_dim,
        text_vocab_size=STUDENT_VOCAB_SIZE,
        text_seq_len=TEXT_SEQ_LEN,
        text_dim=cfg.student.text_dim,
        text_layers=cfg.student.text_layers,
        text_nhead=cfg.student.text_nhead,
        text_ff_dim_multiplier=cfg.student.text_ff_dim_multiplier,
        projection_dim=cfg.student.projection_dim,
    ).to(device)

    # Get student raw dimensions
    with torch.no_grad():
        dummy_images = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)  # [0, 1]
        dummy_text = torch.randint(1, max_valid_token_id_student + 1, (1, TEXT_SEQ_LEN)).to(device)
        _, _, student_vision_raw, student_text_raw = student_model(dummy_images, dummy_text)
        student_vision_dim = student_vision_raw.shape[-1]
        student_text_dim = student_text_raw.shape[-1]

    if is_main_process(rank):
        print(f"Student vision raw dim: {student_vision_dim}")
        print(f"Student text raw dim: {student_text_dim}")

    # Count parameters
    vision_backbone_params = count_parameters(student_model.vision_backbone)
    vision_proj_params = count_parameters(student_model.vision_proj)
    vision_params = vision_backbone_params + vision_proj_params

    # Text model parameters (including position embedding which is a Parameter)
    text_embedding_params = count_parameters(student_model.text_embedding)
    text_transformer_params = count_parameters(student_model.text_transformer)
    text_proj_params = count_parameters(student_model.text_proj)
    text_pos_params = student_model.text_pos_embedding.numel()  # Position embedding is a Parameter
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

    # Get embedding layers (before DDP wrapping)
    student_embedding_layer = student_model.text_embedding
    student_embedding_dim = student_embedding_layer.embedding_dim

    # Wrap model with DDP if using distributed training
    if use_distributed:
        student_model = DDP(
            student_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )
        if is_main_process(rank):
            print("✓ Student model wrapped with DistributedDataParallel")

    # Get teacher embedding layer
    teacher_embedding_layer: nn.Embedding | None = None
    if hasattr(teacher_model, "text_model") and hasattr(teacher_model.text_model, "embeddings"):
        embeddings = teacher_model.text_model.embeddings
        if hasattr(embeddings, "token_embedding"):
            teacher_embedding_layer = cast(nn.Embedding, embeddings.token_embedding)
        elif hasattr(embeddings, "word_embeddings"):
            teacher_embedding_layer = cast(nn.Embedding, embeddings.word_embeddings)

    if teacher_embedding_layer is None:
        if is_main_process(rank):
            print("Warning: Could not find teacher embedding layer")

    # Weight Transfer: One-time initialization of student embeddings from teacher
    if USE_WEIGHT_TRANSFER and teacher_embedding_layer is not None:
        if is_main_process(rank):
            print("\n=== Performing Embedding Weight Transfer ===")
        if hasattr(teacher_embedding_layer, "embedding_dim"):
            teacher_embedding_dim = cast(int, teacher_embedding_layer.embedding_dim)
        else:
            teacher_embedding_dim = student_embedding_dim
        if is_main_process(rank):
            print(f"Student embedding dim: {student_embedding_dim}, Teacher embedding dim: {teacher_embedding_dim}")

        # Create token mapping - use real tokenizers if available, otherwise use dummy
        if is_main_process(rank):
            print("Creating token mapping...")
        if USE_REAL_DATA and student_tokenizer is not None and teacher_tokenizer is not None:
            # Use real tokenizers to find shared tokens
            if is_main_process(rank):
                print("Using real tokenizers to find shared tokens...")
            shared_student_indices, shared_teacher_indices = create_token_mapping(
                teacher_tokenizer=teacher_tokenizer,
                student_tokenizer=student_tokenizer,
                verbose=is_main_process(rank),
            )
        else:
            # Fallback to dummy mapping
            if is_main_process(rank):
                print("Using dummy token mapping (no tokenizers available or using dummy data)...")
            shared_student_indices, shared_teacher_indices = create_dummy_token_mapping(
                student_vocab_size=STUDENT_VOCAB_SIZE,
                teacher_vocab_size=cast(int, teacher_vocab_size),
                overlap_ratio=cfg.embedding.overlap_ratio,
            )
        if is_main_process(rank):
            print(f"Shared tokens: {len(shared_student_indices)} (student) <-> {len(shared_teacher_indices)} (teacher)")

        # Transfer weights (use the embedding layer we already extracted before DDP wrapping)
        transferred_count = transfer_embedding_weights(
            student_embedding_layer=student_embedding_layer,
            teacher_embedding_layer=teacher_embedding_layer,
            shared_student_indices=shared_student_indices,
            shared_teacher_indices=shared_teacher_indices,
            verbose=is_main_process(rank),
        )
        if is_main_process(rank):
            print(f"Successfully transferred {transferred_count} embedding weights!")
            print("No embedding mimicking loss needed - weights already initialized.")
            print("=============================================\n")

    # Create distillation projections (map student raw features to teacher raw features)
    projection_v = nn.Linear(cast(int, student_vision_dim), cast(int, teacher_vision_dim)).to(device)
    projection_t = nn.Linear(cast(int, student_text_dim), cast(int, teacher_text_dim)).to(device)
    logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07)).to(device)
    # Optimizer
    optimizer_params = (
        list(student_model.parameters()) + list(projection_v.parameters()) + list(projection_t.parameters())
    )

    optimizer = torch.optim.AdamW(optimizer_params, lr=LEARNING_RATE)

    # Learning rate scheduler (HuggingFace style with warmup)
    if USE_COSINE_SCHEDULER:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=WARMUP_STEPS,  # Warmup steps
            num_training_steps=MAX_STEPS,  # Total training steps
        )
        if is_main_process(rank):
            print("✓ Cosine learning rate scheduler with warmup initialized (HuggingFace style)")
            print(f"  Initial LR: {LEARNING_RATE}")
            print(f"  Warmup steps: {WARMUP_STEPS}")
            print(f"  Total steps: {MAX_STEPS}")
            print("  Final LR will decay to: ~0 (cosine decay)")
    else:
        scheduler = None

    # Dataset and dataloader
    if USE_REAL_DATA:
        if is_main_process(rank):
            print("\n=== Setting up COCO Dataset ===")
        if student_processor is None:
            raise ValueError("Student processor is required for real data")
        if teacher_processor is None:
            if is_main_process(rank):
                print("Warning: Teacher processor not available. Using fallback preprocessing.")

        # Update student processor augmentation based on split
        use_augmentation_for_split = USE_AUGMENTATION and DATASET_SPLIT == "train"
        if hasattr(student_processor.image_processor, "use_augmentation"):
            student_processor.image_processor.use_augmentation = use_augmentation_for_split
            student_processor.image_processor._build_transform()

        dataset = COCOCaptionDataset(
            split=DATASET_SPLIT,
            image_size=IMAGE_SIZE,
            student_processor=student_processor,
            teacher_processor=teacher_processor,
            max_seq_len=TEXT_SEQ_LEN,
            cache_dir=DATASET_CACHE_DIR,
            use_augmentation=use_augmentation_for_split,
            streaming=cfg.dataset.streaming,
        )
        # For IterableDataset, shuffle is handled differently (use buffer_size)
        # Use DistributedSampler if using distributed training
        sampler = None
        if use_distributed:
            # For IterableDataset, we can't use DistributedSampler directly
            # Instead, we'll handle sharding in the dataset or use a workaround
            # For now, we'll skip the sampler for IterableDataset
            if not cfg.dataset.streaming:
                sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

        # For streaming IterableDataset, num_workers is limited by dataset shards
        # HuggingFace streaming datasets automatically shard, and num_workers cannot exceed num_shards
        # In distributed training, each process gets one shard, so num_workers should be <= 1
        # For non-streaming or when we have more control, we can use more workers
        if cfg.dataset.streaming:
            # For streaming: limit num_workers to 1 per process to avoid shard conflicts
            # Each distributed process will get its own shard automatically
            effective_num_workers = min(DATASET_NUM_WORKERS, 1) if use_distributed else DATASET_NUM_WORKERS
            if is_main_process(rank) and DATASET_NUM_WORKERS > effective_num_workers:
                print(
                    f"Warning: num_workers={DATASET_NUM_WORKERS} reduced to {effective_num_workers} "
                    f"for streaming IterableDataset in distributed training"
                )
        else:
            effective_num_workers = DATASET_NUM_WORKERS

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False if (use_distributed or cfg.dataset.streaming) else True,
            sampler=sampler,
            collate_fn=collate_coco_batch,
            num_workers=effective_num_workers,
            pin_memory=True if (torch.cuda.is_available() and device.type == "cuda") else False,
        )
        # Print dataset info
        if is_main_process(rank):
            if cfg.dataset.streaming:
                print("✓ COCO dataset loaded (streaming mode - size unknown)")
            else:
                print(f"✓ COCO dataset loaded: {len(dataset)} samples")
            print("=" * 30 + "\n")
    else:
        if is_main_process(rank):
            print("\n=== Using Dummy Dataset ===")
        dataset = DummySiglipDataset(
            num_samples=10000,
            img_size=IMAGE_SIZE,
            seq_len=TEXT_SEQ_LEN,
            vocab_size=STUDENT_VOCAB_SIZE,  # Use student vocab size
            max_token_id=max_valid_token_id_student,  # Use student vocab range
        )
        # Use DistributedSampler for dummy dataset if using distributed training
        sampler = None
        if use_distributed:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False if use_distributed else True,
            sampler=sampler,
            num_workers=DATASET_NUM_WORKERS,
        )
        if is_main_process(rank):
            print(f"✓ Dummy dataset created: {len(dataset)} samples")
            print("=" * 30 + "\n")

    # Training loop with max_steps
    # Get the underlying model from DDP wrapper if using distributed training
    if use_distributed:
        model_to_train = cast(nn.Module, student_model.module)
    else:
        model_to_train = student_model
    model_to_train.train()
    if is_main_process(rank):
        print(f"\nStarting training for {MAX_STEPS} steps...")
        print("Note: If dataset ends before max_steps, it will cycle back and continue.")

    # Initialize early stopping
    best_monitor_value = None
    best_step = 0
    patience_counter = 0
    best_model_state = None
    best_projection_v_state = None
    best_projection_t_state = None
    best_logit_scale = None

    if early_stopping_enabled and is_main_process(rank):
        print("\n=== Early Stopping Enabled ===")
        print(f"Monitor: {early_stopping_monitor}")
        print(f"Mode: {early_stopping_mode}")
        print(f"Patience: {early_stopping_patience} steps")
        print(f"Min delta: {early_stopping_min_delta}")
        print(f"Restore best weights: {restore_best_weights}")
        print("=" * 30 + "\n")

    # Create an iterator that can cycle through the dataloader
    step = 0
    dataloader_iter = iter(dataloader)

    # Progress bar for steps (only on main process)
    pbar = tqdm(total=MAX_STEPS, desc="Training", disable=not is_main_process(rank))

    while step < MAX_STEPS:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            # Dataset exhausted, cycle back to the beginning
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
            if use_distributed and sampler is not None:
                sampler.set_epoch(step)  # Set epoch for DistributedSampler
            if is_main_process(rank):
                print(f"\nDataset cycle: Restarting from beginning (step {step}/{MAX_STEPS})")

        if USE_REAL_DATA:
            # COCO dataset returns a dict
            student_images = batch["student_images"].to(device)
            teacher_images = batch["teacher_images"].to(device)
            student_text_ids = batch["student_text_ids"].to(device)
            teacher_text_ids = batch["teacher_text_ids"].to(device)
        else:
            # Dummy dataset returns tuple
            images, text_ids_student = batch
            student_images = images.to(device)
            teacher_images = images.to(device)  # Use same images for dummy data
            student_text_ids = text_ids_student.to(device)
            # Clamp to ensure they're within teacher vocab range
            teacher_text_ids = torch.clamp(student_text_ids, max=max_valid_token_id_teacher)

        optimizer.zero_grad()

        # Teacher forward (frozen)
        with torch.no_grad():
            teacher_outputs = teacher_model(
                pixel_values=teacher_images,
                input_ids=teacher_text_ids,
                output_hidden_states=True,
            )

            teacher_image_features = teacher_outputs.image_embeds
            teacher_text_features = teacher_outputs.text_embeds

            # Get raw features from teacher
            teacher_vision_outputs = teacher_model.vision_model(pixel_values=teacher_images)
            teacher_text_outputs = teacher_model.text_model(input_ids=teacher_text_ids)

            teacher_vision_raw = teacher_vision_outputs.last_hidden_state[:, 0, :]  # CLS token
            teacher_text_raw = teacher_text_outputs.last_hidden_state[:, 0, :]  # CLS token

        # Student forward (uses student vocab token IDs)
        (
            student_image_features,
            student_text_features,
            student_vision_raw,
            student_text_raw,
        ) = model_to_train(student_images, student_text_ids)

        # Compute loss
        loss, loss_dict = compute_total_loss(
            student_image_features=student_image_features,
            student_text_features=student_text_features,
            teacher_image_features=teacher_image_features,
            teacher_text_features=teacher_text_features,
            teacher_vision_raw=teacher_vision_raw,
            teacher_text_raw=teacher_text_raw,
            student_vision_raw=student_vision_raw,
            student_text_raw=student_text_raw,
            projection_v=projection_v,
            projection_t=projection_t,
            logit_scale=logit_scale,
            lambda_cmd=LAMBDA_CMD,
            lambda_umd=LAMBDA_UMD,
            temperature=TEMPERATURE,
        )

        # Backward
        loss.backward()
        optimizer.step()

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        step += 1

        # Compute evaluation metrics periodically
        eval_metrics = {}
        if step % eval_every_n_steps == 0:
            with torch.no_grad():
                eval_metrics = compute_evaluation_metrics(
                    student_image_features=student_image_features,
                    student_text_features=student_text_features,
                    teacher_image_features=teacher_image_features,
                    teacher_text_features=teacher_text_features,
                    logit_scale=logit_scale,
                    projection_v=projection_v,
                    projection_t=projection_t,
                    student_vision_raw=student_vision_raw,
                    student_text_raw=student_text_raw,
                    teacher_vision_raw=teacher_vision_raw,
                    teacher_text_raw=teacher_text_raw,
                )

        # Early stopping check (check every step, but only save best model periodically)
        should_stop = False
        if early_stopping_enabled:
            # Get the monitored value from loss_dict or eval_metrics
            if early_stopping_monitor in loss_dict:
                monitor_value = loss_dict[early_stopping_monitor]
            elif early_stopping_monitor in eval_metrics:
                monitor_value = eval_metrics[early_stopping_monitor]
            else:
                # Default to weighted loss if monitor key not found
                monitor_value = loss_dict.get("weighted", float("inf"))
                if is_main_process(rank) and step == 1:
                    print(f"Warning: Monitor key '{early_stopping_monitor}' not found, using 'weighted' loss instead")

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
                best_step = step
                patience_counter = 0
                # Save best model state (only on main process)
                if is_main_process(rank):
                    best_model_state = model_to_train.state_dict().copy()
                    best_projection_v_state = projection_v.state_dict().copy()
                    best_projection_t_state = projection_t.state_dict().copy()
                    best_logit_scale = logit_scale.data.clone()
                    if step % log_every_n_steps == 0:
                        print(f"✓ New best {early_stopping_monitor}: {monitor_value:.6f} (step {step})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    should_stop = True
                    if is_main_process(rank):
                        print(f"\n{'=' * 70}")
                        print("Early stopping triggered!")
                        print(f"  Best {early_stopping_monitor}: {best_monitor_value:.6f} at step {best_step}")
                        print(f"  Current {early_stopping_monitor}: {monitor_value:.6f} at step {step}")
                        print(f"  No improvement for {patience_counter} steps (patience: {early_stopping_patience})")
                        print(f"{'=' * 70}\n")

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        log_dict = {**loss_dict, **eval_metrics, "step": step, "total": MAX_STEPS, "lr": f"{current_lr:.2e}"}
        pbar.set_postfix(log_dict)
        pbar.update(1)

        # Log to WandB
        if wandb_enabled:
            wandb_log = {
                "train/loss": loss_dict["weighted"],
                "train/loss_siglip": loss_dict["siglip"],
                "train/loss_cmd": loss_dict["cmd"],
                "train/loss_umd": loss_dict["umd"],
                "train/learning_rate": current_lr,
                "train/step": step,
            }
            # Add evaluation metrics if available
            if eval_metrics:
                wandb_log.update({f"eval/{k}": v for k, v in eval_metrics.items()})
            wandb.log(wandb_log, step=step)

        if step % log_every_n_steps == 0 and is_main_process(rank):
            log_message = f"\nStep {step}/{MAX_STEPS}: Loss: {loss_dict} | LR: {current_lr:.2e}"
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
                log_message += (
                    f" | Best {early_stopping_monitor}: {best_monitor_value:.6f} (step {best_step}) "
                    f"| Patience: {patience_counter}/{early_stopping_patience}"
                )
            print(log_message)

        # Check early stopping
        if should_stop:
            break

    pbar.close()

    # Restore best weights if early stopping was enabled and triggered
    if early_stopping_enabled and should_stop and restore_best_weights and is_main_process(rank):
        if (
            best_model_state is not None
            and best_projection_v_state is not None
            and best_projection_t_state is not None
            and best_logit_scale is not None
        ):
            print(f"\nRestoring best model weights from step {best_step}...")
            model_to_train.load_state_dict(best_model_state)
            projection_v.load_state_dict(best_projection_v_state)
            projection_t.load_state_dict(best_projection_t_state)
            logit_scale.data = best_logit_scale
            print(f"✓ Best weights restored (best {early_stopping_monitor}: {best_monitor_value:.6f})")

    if is_main_process(rank):
        if should_stop:
            print(f"\nTraining stopped early after {step} steps (best at step {best_step})!")
        else:
            print(f"\nTraining completed after {step} steps!")

    # Log final metrics to WandB
    if wandb_enabled:
        wandb.summary["final_step"] = step
        wandb.summary["total_steps"] = MAX_STEPS

    # Save checkpoint only on main process
    if is_main_process(rank):
        print(f"Saving model to {checkpoint_path}...")
        # Get state dict from DDP model if needed
        # Use best model state if early stopping was enabled and best weights were restored
        if (
            early_stopping_enabled
            and restore_best_weights
            and best_model_state is not None
            and best_projection_v_state is not None
            and best_projection_t_state is not None
            and best_logit_scale is not None
        ):
            model_state_dict = best_model_state
            projection_v_state = best_projection_v_state
            projection_t_state = best_projection_t_state
            logit_scale_data = best_logit_scale
            print(
                f"  Saving best model from step {best_step} (best {early_stopping_monitor}: {best_monitor_value:.6f})"
            )
        else:
            model_state_dict = model_to_train.state_dict()
            projection_v_state = projection_v.state_dict()
            projection_t_state = projection_t.state_dict()
            logit_scale_data = logit_scale.data

        checkpoint_dict = {
            "student_model": model_state_dict,
            "projection_v": projection_v_state,
            "projection_t": projection_t_state,
            "logit_scale": logit_scale_data,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        if early_stopping_enabled:
            checkpoint_dict["best_step"] = best_step
            checkpoint_dict["best_monitor_value"] = best_monitor_value
            checkpoint_dict["best_monitor_key"] = early_stopping_monitor

        torch.save(checkpoint_dict, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

        # Save configuration to output directory
        config_save_path = output_dir / "config.yaml"
        OmegaConf.save(cfg, config_save_path)
        print(f"Configuration saved to {config_save_path}")

        # Save student processor for inference (Hugging Face style)
        print("\n=== Saving Student Processor for Inference ===")

        if student_processor is not None:
            # Reset augmentation to False for inference
            if hasattr(student_processor.image_processor, "use_augmentation"):
                student_processor.image_processor.use_augmentation = False
                student_processor.image_processor._build_transform()

            processor_save_path = output_dir / "processor"
            try:
                student_processor.save_pretrained(str(processor_save_path))
                print(f"✓ Student processor saved to {processor_save_path}")
                print(f"  - Tokenizer: {processor_save_path / 'tokenizer'}")
                print(f"  - Image processor: {processor_save_path / 'image_processor'}")
            except Exception as e:
                print(f"Warning: Could not save student processor: {e}")
        else:
            print("Warning: Student processor is None, skipping processor save.")
            print("  Note: Inference will require manual tokenizer and image processor setup.")

        print("=" * 50)

        # Finish WandB run
        if wandb_enabled:
            wandb.finish()
            print("✓ WandB run completed")

    # Cleanup distributed training
    cleanup_distributed()


if __name__ == "__main__":
    main()  # Hydra decorator handles configuration loading
