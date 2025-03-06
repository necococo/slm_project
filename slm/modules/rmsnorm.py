import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    RMSNorm実装
    Paper: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        """
        Args:
            dim: 正規化する次元サイズ
            eps: 数値安定性のための小さな値
        """
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMS正規化を適用
        
        Args:
            x: 入力テンソル [..., dim]
            
        Returns:
            正規化された出力テンソル
        """
        # 2乗平均の平方根
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 正規化と重みの適用
        return self.scale * (x / rms)
