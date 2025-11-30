"""
Simplified training script for SigLIP distillation with Hydra configuration.
"""

import math
from pathlib import Path
from typing import cast

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
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
    transfer_embedding_weights_dummy,
)
from tinysiglip.fake_dataset import DummySiglipDataset
from tinysiglip.loss import compute_total_loss
from tinysiglip.metrics import compute_evaluation_metrics
from tinysiglip.model import TinySiglipModel
from tinysiglip.processor import TinySiglipProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


def get_teacher_model(model_name="google/siglip2-base-patch16-224") -> SiglipModel:
    """Load teacher SigLIP model."""
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

    # Initialize WandB
    wandb_enabled = cfg.wandb.get("enabled", False) and WANDB_AVAILABLE
    if wandb_enabled:
        wandb_config = cfg.wandb
        run_name = wandb_config.get("name") or f"tinysiglip_{output_dir.name}"
        wandb.init(
            project=wandb_config.get("project", "tinysiglip"),
            entity=wandb_config.get("entity"),
            name=run_name,
            tags=wandb_config.get("tags", []),
            notes=wandb_config.get("notes", ""),
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(output_dir),
        )
        print(f"✓ WandB initialized: {wandb.run.url if wandb.run else 'N/A'}")
    elif cfg.wandb.get("enabled", False) and not WANDB_AVAILABLE:
        print("Warning: WandB is enabled in config but not installed. Install with: pip install wandb")

    # Print configuration
    print("=" * 70)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint will be saved to: {checkpoint_path}")
    print("=" * 70 + "\n")

    # Load teacher model and processor
    print("Loading teacher model...")
    teacher_model = get_teacher_model(TEACHER_MODEL_NAME)
    teacher_config = teacher_model.config

    # Load teacher processor (recommended for proper preprocessing)
    print("Loading teacher processor...")
    try:
        teacher_processor = AutoProcessor.from_pretrained(TEACHER_MODEL_NAME)
        print("✓ Teacher processor loaded")
    except Exception as e:
        print(f"Warning: Could not load teacher processor: {e}")
        teacher_processor = None

    # Load tokenizers and create student processor for real data
    student_tokenizer = None
    student_processor = None
    if USE_REAL_DATA:
        print("Loading tokenizers for real data...")
        teacher_tokenizer = None

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

    print(f"Teacher projection dim: {teacher_projection_dim}")
    print(f"Teacher vision dim: {teacher_vision_dim}")
    print(f"Teacher text dim: {teacher_text_dim}")
    print(f"Teacher vocab size: {teacher_vocab_size}")

    # Determine student vocab size (use configured value or teacher's vocab size)
    if STUDENT_VOCAB_SIZE is None:
        STUDENT_VOCAB_SIZE = cast(int, teacher_vocab_size)
        print(f"\nStudent vocab size: {STUDENT_VOCAB_SIZE} (same as teacher)")
    else:
        print(f"\nStudent vocab size: {STUDENT_VOCAB_SIZE} (English-only)")
        print("Note: In real training, use tokenizers to map text to token IDs for each model")

    # Calculate valid token ID range (for dummy data only)
    # Use the smaller of the two vocab sizes for token generation
    # In real training, you'd use a tokenizer to map between vocabularies
    max_valid_token_id_student = max(1, STUDENT_VOCAB_SIZE - 10)
    max_valid_token_id_teacher = max(1, cast(int, teacher_vocab_size) - 10)

    # Create student model
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

    # Get embedding layers
    student_embedding_layer = student_model.text_embedding
    student_embedding_dim = student_embedding_layer.embedding_dim

    # Get teacher embedding layer
    teacher_embedding_layer: nn.Embedding | None = None
    if hasattr(teacher_model, "text_model") and hasattr(teacher_model.text_model, "embeddings"):
        embeddings = teacher_model.text_model.embeddings
        if hasattr(embeddings, "token_embedding"):
            teacher_embedding_layer = cast(nn.Embedding, embeddings.token_embedding)
        elif hasattr(embeddings, "word_embeddings"):
            teacher_embedding_layer = cast(nn.Embedding, embeddings.word_embeddings)

    if teacher_embedding_layer is None:
        print("Warning: Could not find teacher embedding layer")

    # Weight Transfer: One-time initialization of student embeddings from teacher
    if USE_WEIGHT_TRANSFER and teacher_embedding_layer is not None:
        print("\n=== Performing Embedding Weight Transfer ===")
        if hasattr(teacher_embedding_layer, "embedding_dim"):
            teacher_embedding_dim = cast(int, teacher_embedding_layer.embedding_dim)
        else:
            teacher_embedding_dim = student_embedding_dim
        print(f"Student embedding dim: {student_embedding_dim}, Teacher embedding dim: {teacher_embedding_dim}")

        # Create token mapping for dummy data (or use real tokenizers in production)
        print("Creating token mapping...")
        shared_student_indices, shared_teacher_indices = create_dummy_token_mapping(
            student_vocab_size=STUDENT_VOCAB_SIZE,
            teacher_vocab_size=cast(int, teacher_vocab_size),
            overlap_ratio=cfg.embedding.overlap_ratio,
        )
        print(f"Shared tokens: {len(shared_student_indices)} (student) <-> {len(shared_teacher_indices)} (teacher)")

        # Transfer weights
        transferred_count = transfer_embedding_weights_dummy(
            student_embedding_layer=student_embedding_layer,
            teacher_embedding_layer=teacher_embedding_layer,
            shared_student_indices=shared_student_indices,
            shared_teacher_indices=shared_teacher_indices,
            verbose=True,
        )
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
        print("✓ Cosine learning rate scheduler with warmup initialized (HuggingFace style)")
        print(f"  Initial LR: {LEARNING_RATE}")
        print(f"  Warmup steps: {WARMUP_STEPS}")
        print(f"  Total steps: {MAX_STEPS}")
        print("  Final LR will decay to: ~0 (cosine decay)")
    else:
        scheduler = None

    # Dataset and dataloader
    if USE_REAL_DATA:
        print("\n=== Setting up COCO Dataset ===")
        if student_processor is None:
            raise ValueError("Student processor is required for real data")
        if teacher_processor is None:
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
        # num_workers=0 for IterableDataset to avoid issues with streaming
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,  # IterableDataset doesn't support shuffle, use buffer_size if needed
            collate_fn=collate_coco_batch,
            num_workers=0,  # Set to 0 for IterableDataset with streaming
            pin_memory=True if torch.cuda.is_available() else False,
        )
        # Note: IterableDataset doesn't support len()
        print("✓ COCO dataset loaded (streaming mode - size unknown)")
        print("=" * 30 + "\n")
    else:
        print("\n=== Using Dummy Dataset ===")
        dataset = DummySiglipDataset(
            num_samples=10000,
            img_size=IMAGE_SIZE,
            seq_len=TEXT_SEQ_LEN,
            vocab_size=STUDENT_VOCAB_SIZE,  # Use student vocab size
            max_token_id=max_valid_token_id_student,  # Use student vocab range
        )
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"✓ Dummy dataset created: {len(dataset)} samples")
        print("=" * 30 + "\n")

    # Training loop with max_steps
    student_model.train()
    print(f"\nStarting training for {MAX_STEPS} steps...")
    print("Note: If dataset ends before max_steps, it will cycle back and continue.")

    # Create an iterator that can cycle through the dataloader
    step = 0
    dataloader_iter = iter(dataloader)

    # Progress bar for steps
    pbar = tqdm(total=MAX_STEPS, desc="Training")

    while step < MAX_STEPS:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            # Dataset exhausted, cycle back to the beginning
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
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
        ) = student_model(student_images, student_text_ids)

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

        if step % log_every_n_steps == 0:
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
            print(log_message)

    pbar.close()
    print(f"\nTraining completed after {step} steps!")

    # Log final metrics to WandB
    if wandb_enabled:
        wandb.summary["final_step"] = step
        wandb.summary["total_steps"] = MAX_STEPS
    print(f"Saving model to {checkpoint_path}...")
    checkpoint_dict = {
        "student_model": student_model.state_dict(),
        "projection_v": projection_v.state_dict(),
        "projection_t": projection_t.state_dict(),
        "logit_scale": logit_scale.data,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
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


if __name__ == "__main__":
    main()  # Hydra decorator handles configuration loading
