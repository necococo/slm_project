# test_loss.py
import torch
from slm.cce_loss import CCELoss

def test_criterion():
    batch_size = 2
    seq_length = 10
    vocab_size = 100
    # ダミーのロジットとラベル
    dummy_logits = torch.randn(batch_size, seq_length, vocab_size)
    dummy_labels = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_length))
    criterion = CCELoss()
    loss = criterion(dummy_logits, dummy_labels)
    # ロスが正のスカラーとなるはず
    assert loss.item() > 0, "Loss should be positive"