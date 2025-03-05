# wavelayer.py

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from slm.config import ModelConfig

def to_wave_representation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    How:
        x (B,S,D) -> float32で安全に波形変換 => ratio => alpha => (real, imag)
        最終的には戻り値も float32 で返す（呼び出し側で必要に応じて .half() する）。
    """
    # Wave計算だけ float32
    x_32 = x.float()

    B, S, D = x_32.shape
    EPS = 1e-5
    
    # グローバル振幅
    G = torch.sqrt(torch.sum(x_32 * x_32, dim=1) + EPS)  # (B,D)
    G_exp = G.unsqueeze(1).expand(-1, S, -1)  # (B,S,D)
    G_safe = torch.clamp(G_exp, min=EPS)
    
    ratio = x_32 / G_safe
    ratio_clamped = torch.clamp(ratio, -0.99, 0.99)
    
    inside = 1.0 - ratio_clamped**2
    # 負値を ReLU でつぶし + EPS
    inside = F.relu(inside) + EPS

    alpha = torch.atan2(torch.sqrt(inside), ratio_clamped)
    real_part = G_safe * torch.cos(alpha)
    imag_part = G_safe * torch.sin(alpha)
    
    return real_part, imag_part  # float32


class SingleWaveLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout_prob: float = 0.1
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # FeedForward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size * 2)
        )
        self.ffn_norm = nn.LayerNorm(hidden_size * 2)

        # restore => (B,S,2D)->(B,S,D)
        self.restore_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.overlay_dropout = nn.Dropout(dropout_prob)
        self.overlay_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        How:
            - Wave変換を安全にfloat32で実行
            - 最終的な出力を .half() して返す
        """
        B, S, D = x_embedding.shape
        EPS = 1e-5
        
        # NaNチェック & float32化
        x_float = x_embedding.float()
        if torch.isnan(x_float).any():
            print("WARNING: NaN detected in layer input, replacing with 0.0")
            x_float = torch.nan_to_num(x_float, nan=0.0)

        # 1) 文レベルwave
        G_sen_scalar = torch.sqrt(torch.sum(x_float * x_float, dim=(1,2)) + EPS)
        G_sen_expanded = G_sen_scalar.view(B,1,1).expand(-1,S,D)
        G_sen_safe = torch.clamp(G_sen_expanded, min=EPS)
        
        ratio_sen = torch.clamp(x_float / G_sen_safe, -0.99, 0.99)
        inside_sen = F.relu(1.0 - ratio_sen**2) + EPS
        alpha_sen = torch.atan2(torch.sqrt(inside_sen), ratio_sen)
        real_sen = G_sen_safe * torch.cos(alpha_sen)
        imag_sen = G_sen_safe * torch.sin(alpha_sen)

        # 2) 入力wave
        real_in, imag_in = to_wave_representation(x_float)  # float32

        # 3) 干渉 (加算)
        out_real = real_sen + real_in
        out_imag = imag_sen + imag_in

        # amplitude
        amplitude = torch.sqrt(out_real**2 + out_imag**2 + EPS)
        sign = torch.sign(out_real + EPS)
        w_recon = amplitude * sign

        # 再wave
        real_w, imag_w = to_wave_representation(w_recon)

        # cat
        wave_cat = torch.cat([real_w, imag_w], dim=-1)  # (B,S,2D)

        # 4) FF + norm
        ff_out = self.ffn(wave_cat)
        wave_cat = self.ffn_norm(wave_cat + ff_out)

        # 5) restore => residual => dropout => norm
        restored = self.restore_linear(wave_cat)  # (B,S,D)
        # overlay: wave_cat の最初の D を residualとみなす例
        overlay_out = restored + wave_cat[..., :D]

        overlay_out = self.overlay_dropout(overlay_out)
        overlay_out = self.overlay_norm(overlay_out)

        # 最終的に half() で返す (cut_cross_entropy用)
        return overlay_out.half()