import torch
from torch import nn

class RoPEEncoding(nn.Module):
    """
    How:
        (real, imag) の複素ベクトルに RoPE を適用。
        real' = real*cos - imag*sin
        imag' = real*sin + imag*cos

    Why not:
        Wave Network と相性が良い実装方法として、(real,imag) に直接回転を掛ける。
        ここでは hidden_size次元 に対して pos依存の位相を乗せる。
    """

    def __init__(self, hidden_size: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        freq_seq = torch.arange(0, hidden_size, dtype=torch.float32)
        freq_seq = 1.0 / (10000.0 ** (freq_seq / float(hidden_size)))
        pos_ids = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        theta = pos_ids * freq_seq.unsqueeze(0)  # (S, hidden_size)

        self.register_buffer("theta", theta, persistent=False)

    def forward(
        self,
        real: torch.Tensor,
        imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        How:
            (B,S,D) -> RoPE -> (B,S,D)
        """
        B, S, D = real.shape
        if S > self.max_seq_len:
            raise ValueError(f"シーケンス長 {S} が RoPE の {self.max_seq_len} を超過")

        theta_cur = self.theta[:S, :D].to(real.device)
        cos_part = torch.cos(theta_cur)
        sin_part = torch.sin(theta_cur)

        real_out = real * cos_part - imag * sin_part
        imag_out = real * sin_part + imag * cos_part

        return real_out, imag_out