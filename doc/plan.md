
# Project Master Plan: TinySigLIP (SigLIP 2 Distillation)

**Project Goal:** Distill the state-of-the-art **SigLIP 2** vision-language model into a lightweight student model ("TinySigLIP") for efficient image-text retrieval.

## 1. Implementation Strategy & Data Pipeline

### 1.1 Data Preprocessing & Optimization (Critical)

To accelerate training and reduce GPU VRAM usage, we will implement an **offline caching strategy** for the Teacher Model's outputs.

* **Offline Feature Extraction:** Instead of forwarding the huge Teacher model at every training step, we will pre-calculate and cache the embeddings.
  * Iterate through the COCO training set once using the Teacher (SigLIP 2).
  * Store `image_embeds` and `text_embeds` (and optionally `logits`) to disk (recommended: LMDB or HDF5 for fast I/O).
  * **Benefit:** Eliminates the computational overhead of the Teacher during the training loop, allowing for larger batch sizes and faster iterations.

### 1.2 Processor & Tokenizer Consistency

To ensure effective knowledge transfer, the input space for the Student must strictly align with the Teacher.

* **Shared Processor:** The Student **must** use the exact same `AutoProcessor` (Tokenizer + Image Processor) as the Teacher.
  * **Tokenizer:** Use the SigLIP 2 (Gemma-based) tokenizer. **Do not** train a new tokenizer. This ensures the token IDs mapping to semantic meanings are identical.
  * **Image Processor:** Use the same normalization (mean/std), resize ($224\times224$), and crop strategies.

## 2. Model Architecture & Alignment

### 2.1 Student Model Design

* **Vision Encoder:** `timm.create_model('vit_tiny_patch16_224')` (or `vit_small`).
* **Text Encoder:** A compact Transformer Encoder (e.g., 4-6 layers, hidden dim 384) initialized from scratch or a distilled BERT/MiniLM.

### 2.2 Dimension Alignment (Projection)

The raw output dimensions of the Student (e.g., 192 or 384) will not match the Teacher (e.g., 768 or 1152).

* **Projection Head:** We must add a trainable `nn.Linear` projection layer to both the Student's image and text encoders.
* **Target Dimension:** The output of this projection layer must strictly match the Teacher's embedding dimension ($D_{teacher}$) to calculate the Feature Imitation Loss (MSE) effectively.

## 3. Distillation Strategy (Loss Functions)

We employ a multi-objective loss function to transfer specific types of knowledge:

$$
\mathcal{L}_{total} = \lambda_{feat} \mathcal{L}_{feat} + \lambda_{affinity} \mathcal{L}_{affinity} + \lambda_{contrast} \mathcal{L}_{contrast}
$$

1. **Feature Imitation Loss ($\mathcal{L}_{feat}$)**
    * **Type:** Mean Squared Error (MSE).
    * **Input:** Aligned Student Embeddings ($S_{proj}$) vs. Cached Teacher Embeddings ($T$).
    * **Goal:** Force the student to inhabit the same semantic space as the teacher.

2. **Affinity (Relation) Loss ($\mathcal{L}_{affinity}$)**
    * **Type:** KL Divergence.
    * **Input:** Student Similarity Matrix ($S \cdot S^T$) vs. Teacher Similarity Matrix ($T \cdot T^T$).
    * **Goal:** Transfer knowledge of "hard negatives" and relative distances between samples in a batch.

3. **Contrastive Loss ($\mathcal{L}_{contrast}$)**
    * **Type:** Sigmoid Loss (SigLIP style) or InfoNCE.
    * **Input:** Student Logits and Ground Truth Labels.
    * **Goal:** Ensure the student learns the actual task (image-text matching) and doesn't just blindly mimic teacher artifacts.

## 4. Comprehensive Experimental Plan & Ablation Study

This section details the experiments required to validate the distillation method.

### 4.1 Experiment Set A: Distillation Strategy Ablation (Loss Components)

**Goal:** Determine the contribution of each loss component (Feature, Affinity, Contrastive) to the final performance.

* **Setup:** Fix Teacher = **SigLIP 2 Large**; Fix Student = **ViT-Small** (or Tiny).
* **Metrics:** Recall@1 (I2T & T2I).

| Exp ID | Configuration | $\mathcal{L}_{feat}$ (MSE) | $\mathcal{L}_{affinity}$ (KL) | $\mathcal{L}_{contrast}$ (Sigmoid) | Hypothesis / Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A1** | **Baseline (No KD)** | ❌ | ❌ | ✅ | Establishes the lower bound; trains student purely on COCO labels. |
| **A2** | **Feat Only KD** | ✅ | ❌ | ❌ | Tests if mimicking embeddings alone is sufficient (without ground truth labels). |
| **A3** | **Feat + Contrast** | ✅ | ❌ | ✅ | Tests the value of direct embedding alignment combined with task supervision. |
| **A4** | **Affinity + Contrast** | ❌ | ✅ | ✅ | Tests if relational knowledge (soft labels) is better than hard feature constraints. |
| **A5** | **Full Method** | ✅ | ✅ | ✅ | **Proposed Method.** Expected to yield optimal performance by combining all signals. |

### 4.2 Experiment Set B: Scalability Analysis (Teacher vs. Student Size)

**Goal:** Analyze how the capacity of the Teacher and Student affects the distillation efficiency.

* **Setup:** Use the **Full Method (A5)** loss configuration for all runs.

| Exp ID | Teacher Model | Student Model | Parameters (T / S) | Research Question |
| :--- | :--- | :--- | :--- | :--- |
| **B1** | **SigLIP 2 Base** | ViT-Tiny | 86M / ~5M | Can a small teacher effectively improve a tiny student? |
| **B2** | **SigLIP 2 Large** | ViT-Tiny | 303M / ~5M | Does a stronger teacher (Large) significantly boost the *same* tiny student vs. Base? |
| **B3** | **SigLIP 2 Base** | ViT-Small | 86M / ~22M | How does student capacity (Tiny vs Small) impact knowledge absorption from a Base teacher? |
| **B4** | **SigLIP 2 Large** | ViT-Small | 303M / ~22M | **Core Target.** Can we achieve high performance with a reasonably small student and a strong teacher? |
| **B5** | **SigLIP 2 So400M** | ViT-Small | 400M / ~22M | (Optional) Diminishing returns test: Does an even larger teacher help? |

### 4.3 Experiment Set C: Architecture Generalization (Optional)

**Goal:** Verify if the distillation method works across different architectures.

| Exp ID | Teacher Model | Student Model | Purpose |
| :--- | :--- | :--- | :--- |
| **C1** | SigLIP 2 Large | **ResNet-50** | Test distillation on a CNN backbone (~25M params). |
| **C2** | SigLIP 2 Large | **MobileNet** | Test on an extremely efficient architecture. |

---

## 5. Final Report Structure (Expected Outcomes)

Your final report code (`src/report_generator.py`) should be able to auto-generate the following tables based on the experiment logs.

**Table 1: Main Results - Distillation Performance vs. Baseline**
*(Compare the best Distilled Student against the Teacher and the No-KD Student)*

| Model | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | Params |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Teacher (SigLIP 2 Large)** | 70.2% | 88.0% | 96.8% | 50.3% | 74.6% | 303M |
| Student (No KD) | 45.0% | 74.6% | 88.1% | 30.2% | 61.5% | ~22M |
| **Student (TinySigLIP)** | **>60%** | **>85%** | **>93%** | **>40%** | **>70%** | ~22M |

**Table 2: Ablation Study Results (Loss Components)**
*(Populate with results from Experiment Set A)*

| Strategy | I2T R@1 | T2I R@1 | Delta (vs Full) |
| :--- | :--- | :--- | :--- |
| **Full Method** | **62.4%** | **44.1%** | - |
| w/o Feature Loss | 60.3% | 42.0% | -2.1% |
| w/o Affinity Loss | 61.0% | 42.7% | -1.4% |
| w/o Contrastive | 31.5% | 22.4% | -30.9% |

---

## 6. References

[1] Zhai, X., et al. (2025). SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features. arXiv preprint arXiv:2502.14786.

[2] Wu, K., et al. (ICCV 2023). TinyCLIP: CLIP Distillation via Affinity Mimicking and Weight Inheritance.

[3] Yang, H., et al. (CVPR 2024). CLIP-KD: An Empirical Study of CLIP Model Distillation.

[4] Lin, T., et al. (2014). Microsoft COCO: Common Objects in Context.
