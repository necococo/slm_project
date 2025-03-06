import torch
from torch import nn

# class RoPEEncoding(nn.Module):
#     """
#     How:
#         (real, imag) の複素ベクトルに RoPE を適用。
#         real' = real*cos - imag*sin
#         imag' = real*sin + imag*cos

#     Why not:
#         Wave Network と相性が良い実装方法として、(real,imag) に直接回転を掛ける。
#         ここでは hidden_size次元 に対して pos依存の位相を乗せる。
#     """

#     def __init__(self, hidden_size: int, max_seq_len: int = 2048) -> None:
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.max_seq_len = max_seq_len

#         freq_seq = torch.arange(0, hidden_size, dtype=torch.float32)
#         freq_seq = 1.0 / (10000.0 ** (freq_seq / float(hidden_size)))
#         pos_ids = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
#         theta = pos_ids * freq_seq.unsqueeze(0)  # (S, hidden_size)

#         self.register_buffer("theta", theta, persistent=False)

#     def forward(
#         self,
#         real: torch.Tensor,
#         imag: torch.Tensor
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         How:
#             (B,S,D) -> RoPE -> (B,S,D)
#         """
#         B, S, D = real.shape
#         if S > self.max_seq_len:
#             raise ValueError(f"シーケンス長 {S} が RoPE の {self.max_seq_len} を超過")

#         theta_cur = self.theta[:S, :D].to(real.device)
#         cos_part = torch.cos(theta_cur)
#         sin_part = torch.sin(theta_cur)

#         real_out = real * cos_part - imag * sin_part
#         imag_out = real * sin_part + imag * cos_part

#         return real_out, imag_out
    

class RoPEEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) 実装
    参照: https://arxiv.org/abs/2104.09864
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # RoPEで使用するsin/cosテーブルを事前計算
        # dim/2 の長さの周波数リストを作成
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        """
        入力テンソルにRoPEを適用
        Args:
            x: 入力テンソル [batch_size, seq_len, dim]
            seq_dim: シーケンス次元のインデックス
        """
        seq_len = x.shape[seq_dim]
        
        # 位置インデックス [seq_len]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        
        # [seq_len, dim/2] のsinθとcosθの配列を生成
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # [seq_len, dim] に拡張
        freqs = torch.cat((freqs, freqs), dim=-1)
        
        # RoPEの適用
        if seq_dim == 1:
            # [1, seq_len, 1, dim]
            freqs = freqs.unsqueeze(0).unsqueeze(2)
            
            # 偶数と奇数の次元を分割
            x_even = x[:, :, :, ::2]
            x_odd = x[:, :, :, 1::2]
            
            # 複素数回転を適用
            cos, sin = torch.cos(freqs), torch.sin(freqs)
            x_real = x_even * cos - x_odd * sin
            x_imag = x_even * sin + x_odd * cos
            
            # 再結合
            x = torch.stack((x_real, x_imag), dim=-1).flatten(-2)
        
        return x