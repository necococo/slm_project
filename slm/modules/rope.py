import torch
import torch.nn as nn
import math
from typing import Optional

class RoPEEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)の実装
    Paper: https://arxiv.org/abs/2104.09864
    
    引数:
        dim: 埋め込みの次元
        max_seq_len: 最大シーケンス長
        base: RoPEの周波数base
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 周波数の逆数
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力テンソルにRotary Position Embeddingを適用
        
        Args:
            x: 入力テンソル [batch_size, seq_len, 1, dim]
            
        Returns:
            RoPEが適用されたテンソル [batch_size, seq_len, 1, dim]
        """
        seq_len = x.size(1)
        
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0).unsqueeze(2)
        
        # 交互に cos, sin を適用するために偶数・奇数インデックスに分ける
        x_even = x[..., 0::2]  # 偶数インデックス
        x_odd = x[..., 1::2]   # 奇数インデックス
        
        # 回転行列の適用（複素数の乗算と同じ）
        cos = torch.cos(emb)[:,:,:,:x.size(-1)//2]
        sin = torch.sin(emb)[:,:,:,:x.size(-1)//2]
        
        # 実部と虚部への変換（rotaryの論文における回転）
        x_out = torch.zeros_like(x)
        x_out[..., 0::2] = x_even * cos - x_odd * sin
        x_out[..., 1::2] = x_odd * cos + x_even * sin
        
        return x_out