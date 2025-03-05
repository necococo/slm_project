# slm/inference.py
# Why not: 簡単なGreedyサンプリング。QAとしての回答抽出ではない。

import torch
from slm.model import WaveHierarchicalLM

def sample_text(
    model: WaveHierarchicalLM,
    input_ids: torch.Tensor,
    max_len: int = 50,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    How:
        Greedy方式で連続トークンを生成する簡易推論関数。
    """
    model.eval()
    generated = input_ids.clone().to(device)

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(generated)  # (B, seq, vocab)
            next_token_logit = logits[:, -1, :]
            next_token = torch.argmax(next_token_logit, dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=1)

    return generated

