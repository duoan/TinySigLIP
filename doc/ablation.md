## 4. Comprehensive Experimental Plan & Ablation Study

This section details the experiments required to validate the distillation method, strictly following the project proposal[cite: 121, 364].

### 4.1 Experiment Set A: Distillation Strategy Ablation (Loss Components)

**Goal:** Determine the contribution of each loss component (Feature, Affinity, Contrastive) to the final performance[cite: 131, 378].

* **Setup:** Fix Teacher = **SigLIP 2 Large**; Fix Student = **ViT-Small** (or Tiny).
* **Metrics:** Recall@1 (I2T & T2I).

| Exp ID | Configuration | $\mathcal{L}_{feat}$ (MSE) | $\mathcal{L}_{affinity}$ (KL) | $\mathcal{L}_{contrast}$ (Sigmoid) | Hypothesis / Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A1** | **Baseline (No KD)** | ❌ | ❌ | ✅ | Establishes the lower bound; trains student purely on COCO labels[cite: 130, 375]. |
| **A2** | **Feat Only KD** | ✅ | ❌ | ❌ | Tests if mimicking embeddings alone is sufficient [without ground truth labels](cite: 134). |
| **A3** | **Feat + Contrast** | ✅ | ❌ | ✅ | Tests the value of direct embedding alignment combined with task supervision[cite: 132, 379]. |
| **A4** | **Affinity + Contrast** | ❌ | ✅ | ✅ | Tests if relational knowledge (soft labels) is better than hard feature constraints[cite: 133, 380]. |
| **A5** | **Full Method** | ✅ | ✅ | ✅ | **Proposed Method.** Expected to yield optimal performance by combining all signals[cite: 135, 428]. |

### 4.2 Experiment Set B: Scalability Analysis (Teacher vs. Student Size)

**Goal:** Analyze how the capacity of the Teacher and Student affects the distillation efficiency[cite: 122, 365].

* **Setup:** Use the **Full Method (A5)** loss configuration for all runs.

| Exp ID | Teacher Model | Student Model | Parameters (T / S) | Research Question |
| :--- | :--- | :--- | :--- | :--- |
| **B1** | **SigLIP 2 Base** | ViT-Tiny | 86M / ~5M | Can a small teacher effectively improve a tiny student?[cite: 122, 125]. |
| **B2** | **SigLIP 2 Large** | ViT-Tiny | 303M / ~5M | Does a stronger teacher (Large) significantly boost the *same* tiny student vs. Base?[cite: 123]. |
| **B3** | **SigLIP 2 Base** | ViT-Small | 86M / ~22M | How does student capacity (Tiny vs Small) impact knowledge absorption from a Base teacher?[cite: 126, 370]. |
| **B4** | **SigLIP 2 Large** | ViT-Small | 303M / ~22M | **Core Target.** Can we achieve high performance with a reasonably small student and a strong teacher?[cite: 123, 366]. |
| **B5** | **SigLIP 2 So400M** | ViT-Small | 400M / ~22M | (Optional) Diminishing returns test: Does an even larger teacher help?[cite: 122]. |

### 4.3 Experiment Set C: Architecture Generalization (Optional)

**Goal:** Verify if the distillation method works across different architectures [e.g., CNN vs ViT](cite: 369).

| Exp ID | Teacher Model | Student Model | Purpose |
| :--- | :--- | :--- | :--- |
| **C1** | SigLIP 2 Large | **ResNet-50** | Test distillation on a CNN backbone [~25M params](cite: 371). |
| **C2** | SigLIP 2 Large | **MobileNet** | Test on an extreme efficient architecture[cite: 372]. |

---

## 5. Final Report Structure (Expected Outcomes)

Your final report code (`src/report_generator.py`) should be able to auto-generate the following tables based on the experiment logs.

**Table 1: Main Results - Distillation Performance vs. Baseline**
*(Compare the best Distilled Student against the Teacher and the No-KD Student)*[cite: 187, 414].

| Model | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | Params |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Teacher (SigLIP 2 Large)** | 70.2% | 88.0% | 96.8% | 50.3% | 74.6% | 303M |
| Student (No KD) | 45.0% | 74.6% | 88.1% | 30.2% | 61.5% | ~22M |
| **Student (TinySigLIP)** | **>60%** | **>85%** | **>93%** | **>40%** | **>70%** | ~22M |

**Table 2: Ablation Study Results (Loss Components)**
*(Populate with results from Experiment Set A)*[cite: 191, 430].

| Strategy | I2T R@1 | T2I R@1 | Delta (vs Full) |
| :--- | :--- | :--- | :--- |
| **Full Method** | **62.4%** | **44.1%** | - |
| w/o Feature Loss | 60.3% | 42.0% | -2.1% |
| w/o Affinity Loss | 61.0% | 42.7% | -1.4% |
| w/o Contrastive | 31.5% | 22.4% | -30.9% |
