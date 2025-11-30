import torch
from torch.utils.data import Dataset


class DummySiglipDataset(Dataset):
    """
    A dummy dataset that generates random image and token ID tensors.
    Used for demonstrating the training loop without real data.
    """

    def __init__(self, num_samples, img_size, seq_len, vocab_size, max_token_id=None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        # Ensure token IDs are within valid range [1, max_token_id] to avoid special tokens
        self.max_token_id = max_token_id if max_token_id is not None else max(1, vocab_size - 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 图像：(3, H, W) 浮点数，范围 [0, 1]
        # SigLIP 期望像素值在 [0, 1] 范围
        image = torch.rand(3, self.img_size, self.img_size)

        # 文本：(L,) 长整型，模拟 Token ID
        # 随机生成介于 1 到 max_token_id 的 ID（避免 0 可能是 padding）
        token_ids = torch.randint(1, self.max_token_id + 1, (self.seq_len,), dtype=torch.long)

        return image, token_ids
