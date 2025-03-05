# slm/tests/test_train.py
# Why not: 学習フローが最後まで動作するかをテスト。

import pytest
import torch
from torch.utils.data import Dataset

from slm.config import ModelConfig, TrainingConfig
from slm.train import train_mlm_then_diffusion

class DummyDataset(Dataset):
    """
    What:
        テスト用ダミーデータ。 (B,S)のランダムトークンを返す。
    """
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int) -> None:
        super().__init__()
        self.data = []
        for _ in range(num_samples):
            tokens = torch.randint(0, vocab_size, (seq_len,))
            self.data.append({"input_ids": tokens, "labels": tokens})

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

@pytest.mark.parametrize("use_wavelet", [False, True])
def test_train_process(use_wavelet: bool) -> None:
    """
    What:
        MLM -> 拡散Fine-tuning の学習を短いステップで走らせ、
        エラーなく完了するか確認。Wavelet on/off の両方テスト。
    """
    seq_len = 8
    vocab_size = 50
    dataset = DummyDataset(num_samples=16, seq_len=seq_len, vocab_size=vocab_size)

    model_cfg = ModelConfig(
        hidden_size=16,
        num_layers=1,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        use_wavelet=use_wavelet,
        wavelet_name="haar"
    )
    train_cfg = TrainingConfig(
        batch_size=4,
        accumulation_steps=1,
        learning_rate=1e-3,
        max_steps=10,
        mlm_mask_ratio=0.15
    )

    device = torch.device("cpu")
    model = train_mlm_then_diffusion(
        dataset, device, model_cfg, train_cfg,
        mlm_epochs=1,
        diffusion_epochs=1
    )
    assert model is not None

