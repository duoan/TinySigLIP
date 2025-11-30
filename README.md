# TinySigLIP

A SigLIP model distillation implementation based on the TinyCLIP approach, using the `timm` library to construct the student model.

## Overview

TinySigLIP is a knowledge distillation framework for creating compact vision-language models by distilling knowledge from large SigLIP teacher models to smaller student models. The framework supports multiple distillation strategies including cross-modal distillation (CMD), uni-modal distillation (UMD), and embedding layer knowledge transfer.

## Model Architecture

The student model consists of a vision encoder (from `timm`) and a lightweight text encoder:

```mermaid
graph TB
    subgraph "Student Model"
        I[Input Images<br/>B×3×H×W] --> VE[Vision Encoder<br/>timm backbone]
        T[Input Text IDs<br/>B×seq_len] --> TE[Text Embedding<br/>vocab_size×dim]
        TE --> TP[Positional Embedding]
        TP --> TT[Text Transformer<br/>L layers]
        VE --> VP[Vision Projection<br/>Linear]
        TT --> TP2[Text Projection<br/>Linear]
        VP --> NF1[L2 Normalize]
        TP2 --> NF2[L2 Normalize]
        NF1 --> SIM[Similarity Matrix<br/>B×B]
        NF2 --> SIM
    end

    style I fill:#e1f5ff
    style T fill:#e1f5ff
    style VE fill:#fff4e1
    style TE fill:#fff4e1
    style SIM fill:#ffe1f5
```

## Training Approach

The distillation process uses multiple loss components to transfer knowledge from teacher to student:

```mermaid
graph LR
    subgraph "Teacher Model"
        TI[Teacher Images] --> TF[Teacher Features]
        TT[Teacher Text] --> TF
    end

    subgraph "Student Model"
        SI[Student Images] --> SF[Student Features]
        ST[Student Text] --> SF
    end

    TF --> L1[SigLIP Loss<br/>Binary Cross-Entropy]
    SF --> L1

    TF --> L2[CMD Loss<br/>KL Divergence]
    SF --> L2

    TF --> L3[UMD Loss<br/>MSE on Features]
    SF --> L3

    L1 --> TL[Total Loss]
    L2 --> TL
    L3 --> TL

    TL --> OPT[Optimizer]

    style TF fill:#ffe1f5
    style SF fill:#e1f5ff
    style TL fill:#fff4e1
```

### Loss Components

1. **SigLIP Loss** ($\mathcal{L}_{SigLIP}$): Binary cross-entropy with sigmoid activation for contrastive learning
   $$\mathcal{L}_{SigLIP} = \frac{1}{2}\left[\text{BCE}(\sigma(\mathbf{S}_{I2T}), \mathbf{Y}) + \text{BCE}(\sigma(\mathbf{S}_{T2I}), \mathbf{Y})\right]$$

2. **Cross-Modal Distillation (CMD)** ($\mathcal{L}_{CMD}$): KL divergence between teacher and student similarity distributions
   $$\mathcal{L}_{CMD} = \text{KL}\left(P_T(\mathbf{S}_T) \| P_S(\mathbf{S}_S)\right)$$

3. **Uni-Modal Distillation (UMD)** ($\mathcal{L}_{UMD}$): MSE loss on normalized features from vision and text encoders
   $$\mathcal{L}_{UMD} = \frac{1}{2}\left[\text{MSE}(\mathbf{f}_V^T, \mathbf{f}_V^S) + \text{MSE}(\mathbf{f}_T^T, \mathbf{f}_T^S)\right]$$

4. **Total Loss**:
   $$\mathcal{L}_{total} = \lambda_{SigLIP} \cdot \mathcal{L}_{SigLIP} + \lambda_{CMD} \cdot \mathcal{L}_{CMD} + \lambda_{UMD} \cdot \mathcal{L}_{UMD}$$

## Project Structure

- `tinysiglip/model.py`: Student model definition (uses `timm` for vision encoder)
- `tinysiglip/loss.py`: Distillation loss functions (SigLIP loss + CMD + UMD + Embedding Mimicking)
- `tinysiglip/embedding_distillation.py`: Token Embedding Layer distillation utilities
- `tinysiglip/coco_dataset.py`: COCO dataset implementation
- `tinysiglip/fake_dataset.py`: Dummy dataset for testing
- `tinysiglip/processor.py`: Data preprocessing utilities
- `tinysiglip/metrics.py`: Evaluation metrics
- `train.py`: Training script with Hydra configuration

## Usage

### Training

```bash
python train.py
```

The training script uses Hydra for configuration management. Modify `config/config.yaml` to adjust hyperparameters.

### Evaluation

Evaluate your trained model on ImageNet-1k zero-shot classification:

```bash
python eval_imagenet1k.py \
    --imagenet-val /path/to/imagenet/val \
    --resume /path/to/checkpoint.pt \
    --batch-size 32 \
    --num-workers 4
```

**Arguments:**
- `--imagenet-val`: Path to ImageNet validation set directory
- `--resume`: Path to checkpoint file (`.pt` file saved during training)
- `--batch-size`: Batch size for evaluation (default: 32)
- `--num-workers`: Number of data loading workers (default: 4)
- `--device`: Device to use (default: `cuda`)
- `--logit-scale`: Optional logit scale (temperature). If not specified, uses value from checkpoint.

**Example:**
```bash
# Evaluate a trained model
python eval_imagenet1k.py \
    --imagenet-val ./ImageNet \
    --resume ./outputs/2025-11-29_20-15-10/checkpoint.pt \
    --batch-size 64
```

The evaluation script will:
1. Load the checkpoint and restore model configuration
2. Load or create the processor from checkpoint directory
3. Generate text prompts for all 1000 ImageNet classes (e.g., "a photo of a {class_name}")
4. Compute text features for all classes
5. Evaluate on ImageNet validation set and report Top-1 and Top-5 accuracy

**Note**: The checkpoint directory should contain a `processor/` subdirectory (saved automatically during training) for proper text tokenization. If not available, the script will attempt to create a processor from the checkpoint configuration.

## Key Features

1. **Timm-based Student Model**: No need for custom model architecture; directly use pre-trained models from `timm`
2. **SigLIP Distillation**:
   - SigLIP loss (sigmoid-based contrastive loss)
   - Cross-modal distillation loss (CMD)
   - Uni-modal feature distillation loss (UMD)
   - **Token Embedding Layer Distillation** (Embedding Mimicking Loss)
3. **Vocabulary Optimization**: Support for student models using smaller English-specific vocabularies (e.g., 32K vs 256K), significantly reducing parameters
4. **Simplified Design**: Clear code structure, easy to understand and modify

## Configuration

You can modify the following in `config/config.yaml`:

- **Teacher Model**: `teacher.model_name` (default: `google/siglip-base-patch16-224`)
- **Student Vision Model**: `student.vision_model_name` (timm model name, e.g., `vit_tiny_patch16_224`)
- **Student Vocabulary Size**: `student.vocab_size` (default: 32000, for English-only models)
  - Can be set smaller than teacher model to save parameters
  - Common sizes: 32000 (English BPE), 50257 (GPT-2), 49152 (English CLIP)
  - Set to `null` to use the same vocabulary size as teacher model
- **Training Hyperparameters**: Batch size, learning rate, etc.

## Different Vocabulary Sizes

The student model can use a different vocabulary size than the teacher model, which is useful for creating smaller English-specific models.

**Note**: In real applications (with actual text data), you need to:
1. Create/select a tokenizer for the student model (e.g., sentencepiece, BPE)
2. Use different tokenizers to convert the same text to different token IDs
3. Student model uses student tokenizer's token IDs
4. Teacher model uses teacher tokenizer's token IDs

The current code uses dummy data. For demonstration and testing, both models use the same token ID range.

## Token Embedding Layer Distillation / Weight Transfer

When the student model uses a smaller vocabulary than the teacher model (e.g., 32K vs 256K), there are two methods to transfer knowledge:

### Method 1: Weight Transfer (Recommended) ⭐

**Core Advantage**: Zero runtime overhead, one-time initialization

- Before training starts, directly copy weights of shared tokens from teacher model's embedding layer to student model
- No need to compute additional loss terms in the training loop
- Achieves maximum parameter compression

**Implementation**:
- Set `USE_WEIGHT_TRANSFER = True` (default)
- Weights are automatically transferred after model initialization
- Remaining non-shared tokens are randomly initialized and learned during training

**Parameter Savings**: Using a 32K vocabulary compared to 256K can save approximately **86M parameters** (in the embedding layer)

### Method 2: Embedding Mimicking Loss

**Core Idea**:
- Find **shared tokens** in student and teacher vocabularies
- During training, continuously make student model's embeddings for these shared tokens mimic teacher model's embeddings
- Implemented via MSE loss: $\mathcal{L}_{Emb} = \text{MSE}(\text{Emb}_S(\text{shared tokens}), \text{Emb}_T(\text{shared tokens}))$

**Implementation**:
- Set `USE_WEIGHT_TRANSFER = False`
- Weight controlled by `LAMBDA_EMBEDDING` (default 0.0, as weight transfer is recommended)
- Continuously computed during training

### Using Real Tokenizers

In real applications, you can use actual tokenizers for weight transfer:

```python
from tinysiglip.embedding_distillation import transfer_embedding_weights
from transformers import AutoTokenizer

# Load tokenizers
student_tokenizer = AutoTokenizer.from_pretrained("your-student-tokenizer")
teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)

# Execute weight transfer
transfer_embedding_weights(
    student_embedding_layer=student_model.text_embedding,
    teacher_embedding_layer=teacher_model.text_model.embeddings.token_embedding,
    student_tokenizer=student_tokenizer,
    teacher_tokenizer=teacher_tokenizer,
)
```

## Ablation Study

<!-- TODO: Fill in ablation study results -->

### Experimental Setup

- **Teacher Model**: [To be filled]
- **Student Model**: [To be filled]
- **Dataset**: [To be filled]
- **Evaluation Metrics**: [To be filled]

### Results

| Configuration | SigLIP Loss | CMD Loss | UMD Loss | Embedding Transfer | Image-Text Retrieval (R@1) | Text-Image Retrieval (R@1) | Parameters |
|--------------|-------------|----------|----------|-------------------|---------------------------|---------------------------|------------|
| Baseline (SigLIP only) | ✓ | ✗ | ✗ | ✗ | [To be filled] | [To be filled] | [To be filled] |
| + CMD | ✓ | ✓ | ✗ | ✗ | [To be filled] | [To be filled] | [To be filled] |
| + UMD | ✓ | ✓ | ✓ | ✗ | [To be filled] | [To be filled] | [To be filled] |
| + Embedding Transfer | ✓ | ✓ | ✓ | ✓ | [To be filled] | [To be filled] | [To be filled] |

### Analysis

[To be filled: Analysis of ablation study results, including:
- Impact of each loss component
- Effectiveness of embedding weight transfer vs. mimicking loss
- Trade-offs between model size and performance
- Comparison with other distillation methods]

## References

### Core Papers

1. **SigLIP**: Zhai, X., Mustafa, B., Kolesnikov, A., & Beyer, L. (2023). Sigmoid Loss for Language Image Pre-Training. *arXiv preprint arXiv:2303.15343*. [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)

2. **TinyCLIP**: Wu, H., et al. (2023). TinyCLIP: CLIP Distillation via Affinity Mimicking and Weight Inheritance. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*. [arXiv:2301.12562](https://arxiv.org/abs/2301.12562)

3. **CLIP**: Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *International Conference on Machine Learning (ICML)*. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

### Knowledge Distillation

4. **Knowledge Distillation**: Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *arXiv preprint arXiv:1503.02531*. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)

5. **Feature Distillation**: Romero, A., et al. (2014). FitNets: Hints for Thin Deep Nets. *arXiv preprint arXiv:1412.6550*. [arXiv:1412.6550](https://arxiv.org/abs/1412.6550)

### Vision-Language Models

6. **ALIGN**: Jia, C., et al. (2021). Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision. *International Conference on Machine Learning (ICML)*. [arXiv:2102.05918](https://arxiv.org/abs/2102.05918)

7. **BLIP**: Li, J., et al. (2022). BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. *International Conference on Machine Learning (ICML)*. [arXiv:2201.12086](https://arxiv.org/abs/2201.12086)

### Model Compression

8. **Model Compression Survey**: Choudhary, T., et al. (2020). A Comprehensive Survey on Model Compression and Acceleration. *Artificial Intelligence Review*, 53(7), 5113-5155. [Link](https://link.springer.com/article/10.1007/s10462-020-09838-1)

## License

See [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{tinysiglip2024,
  title={TinySigLIP: SigLIP Model Distillation with Timm-based Student Architecture},
  author={[Your Name]},
  year={2024},
  howpublished={\url{https://github.com/yourusername/TinySigLIP}}
}
```
