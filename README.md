# TinySigLIP

基于 TinyCLIP 方法的 SigLIP 模型蒸馏实现，使用 timm 库构建学生模型。

## 结构

- `tinysiglip/model.py`: 学生模型定义（使用 timm 构建视觉编码器）
- `tinysiglip/loss.py`: 蒸馏损失函数（SigLIP loss + CMD + UMD + Embedding Mimicking）
- `tinysiglip/embedding_distillation.py`: Token Embedding Layer 蒸馏工具
- `tinysiglip/data.py`: 数据集定义
- `train.py`: 训练脚本

## 使用方法

```bash
python train.py
```

## 主要特点

1. **使用 timm 构建学生模型**: 无需自定义模型结构，直接使用 timm 中的预训练模型
2. **SigLIP 蒸馏**:
   - SigLIP 损失（sigmoid-based contrastive loss）
   - 跨模态蒸馏损失（CMD）
   - 单模态特征蒸馏损失（UMD）
   - **Token Embedding Layer 蒸馏**（Embedding Mimicking Loss）
3. **词汇表优化**: 支持学生模型使用更小的英语专用词汇表（如 32K vs 256K），大幅减少参数
4. **简化设计**: 代码结构清晰，易于理解和修改

## 配置

可以在 `train.py` 中修改：
- 教师模型：`TEACHER_MODEL_NAME`
- 学生视觉模型：`vision_model_name`（timm 模型名）
- 学生词汇表大小：`STUDENT_VOCAB_SIZE`（默认 32000，可用于仅英语模型）
  - 可以设置比教师模型更小的词汇表以节省参数
  - 常见大小：32000 (English BPE), 50257 (GPT-2), 49152 (English CLIP)
  - 设置为 `None` 则使用与教师模型相同的词汇表大小
- 批次大小、学习率等超参数

## 不同词汇表大小

学生模型可以使用与教师模型不同的词汇表大小，这对于创建更小的英语专用模型很有用。

**注意**：在实际应用中（使用真实文本数据），你需要：
1. 为学生模型创建/选择一个 tokenizer（如 sentencepiece, BPE）
2. 使用不同的 tokenizer 将同一段文本转换为不同的 token IDs
3. 学生模型使用学生 tokenizer 的 token IDs
4. 教师模型使用教师 tokenizer 的 token IDs

当前代码使用虚拟数据，对于演示和测试，两个模型使用相同的 token ID 范围。

## Token Embedding Layer 蒸馏 / 权重继承

当学生模型使用比教师模型更小的词汇表时（如 32K vs 256K），有两种方法传递知识：

### 方法 1: 权重继承（推荐）⭐

**核心优势**：零运行时开销，一次初始化即可

- 在训练开始前，直接从教师模型的 embedding 层复制共享 token 的权重到学生模型
- 无需在训练循环中计算额外的损失项
- 实现最大程度的参数压缩

**实现**：
- 设置 `USE_WEIGHT_TRANSFER = True`（默认）
- 权重会在模型初始化后自动转移
- 剩余的非共享 token 随机初始化，在训练过程中学习

**参数节省**：使用 32K 词汇表相比 256K 可以节省约 **86M 参数**（在 embedding 层）

### 方法 2: Embedding Mimicking Loss

**核心思想**：
- 找到学生和教师词汇表中的**共享 token**
- 在训练过程中持续让学生模型对这些共享 token 的 embedding 模仿教师模型的 embedding
- 通过 MSE 损失来实现：$\mathcal{L}_{Emb} = \text{MSE}(\text{Emb}_S(\text{共享Tokens}), \text{Emb}_T(\text{共享Tokens}))$

**实现**：
- 设置 `USE_WEIGHT_TRANSFER = False`
- 权重由 `LAMBDA_EMBEDDING` 控制（默认 0.0，因为推荐使用权重转移）
- 在训练过程中持续计算损失

### 使用真实 Tokenizer

在实际应用中，可以使用真实的 tokenizer 进行权重转移：

```python
from tinysiglip.embedding_distillation import transfer_embedding_weights
from transformers import AutoTokenizer

# 加载 tokenizers
student_tokenizer = AutoTokenizer.from_pretrained("your-student-tokenizer")
teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)

# 执行权重转移
transfer_embedding_weights(
    student_embedding_layer=student_model.text_embedding,
    teacher_embedding_layer=teacher_model.text_model.embeddings.token_embedding,
    student_tokenizer=student_tokenizer,
    teacher_tokenizer=teacher_tokenizer,
)
```
