"""
Root Mean Square Layer Normalization (RMSNorm)の実装
LayerNormの効率的な代替として、平均の計算を省略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    参考文献: "Root Mean Square Layer Normalization" https://arxiv.org/abs/1910.07467
    
    通常のLayerNormと比較して:
    1. 平均の計算を省略することで、計算効率が向上
    2. ニューラルネットワークの学習安定性が向上
    3. より速いトレーニング収束
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6, elementwise_affine: bool = True):
        """
        Args:
            hidden_size: 正規化する特徴の次元数
            eps: 数値安定性のための小さな値
            elementwise_affine: 学習可能なアフィン変換パラメータを使用するかどうか
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter('weight', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSNormの順伝播
        
        Args:
            x: 入力テンソル [*, hidden_size] (任意のバッチ次元 + 特徴次元)
            
        Returns:
            正規化されたテンソル
        """
        # 二乗平均平方根の計算
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        
        # 学習可能なスケーリング
        if self.elementwise_affine:
            x_normed = self.weight * x_normed
            
        return x_normed
    
    def extra_repr(self) -> str:
        """文字列表現"""
        return f"{self.hidden_size}, eps={self.eps}, " \
               f"elementwise_affine={self.elementwise_affine}"


class ScaleNorm(nn.Module):
    """
    スケール正規化 (ScaleNorm)
    
    RMSNormのさらに簡略化されたバージョンで、
    単一の学習可能なスケールパラメータのみを持ちます。
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Args:
            hidden_size: 正規化する特徴の次元数（実際には使用しないが、APIの一貫性のために）
            eps: 数値安定性のための小さな値
        """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(hidden_size ** 0.5, dtype=torch.float32))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ScaleNormの順伝播
        
        Args:
            x: 入力テンソル [*, hidden_size]
            
        Returns:
            正規化されたテンソル
        """
        norm = torch.mean(x * x, dim=-1, keepdim=True).sqrt()
        x = self.scale * x / (norm + self.eps)
        return x
