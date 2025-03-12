# slm/diffusion.py
# Why not: Large Language Diffusion Models / LlaDA論文を簡易再現する例

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class SimpleTextDiffusion(nn.Module):
    """
    テキスト用のシンプルな拡散モデル
    マスキングを使った非常にシンプルな実装
    """
    def __init__(
        self, 
        timesteps: int = 10, 
        mask_token_id: int = 4,
        vocab_size: int = 30000,
        beta_schedule: str = "linear"
    ):
        """
        Args:
            timesteps: 拡散ステップ数
            mask_token_id: マスクトークンID
            vocab_size: 語彙サイズ
            beta_schedule: ノイズスケジュール ("linear", "cosine")
        """
        super().__init__()
        self.timesteps = timesteps
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        
        # ノイズスケジュール - より緩やかな開始
        if beta_schedule == "linear":
            # 線形スケジュール - より小さな開始値でより緩やかに開始
            betas = torch.linspace(0.05, 0.8, timesteps)  # 小さな値から始めて、最大値も抑える
        elif beta_schedule == "cosine":
            # コサインスケジュール
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.tensor(torch.pi / 2)) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0, 0.8)  # 最大値を0.8に制限
        elif beta_schedule == "quadratic":
            # 二次関数的スケジュール - より滑らかな増加
            betas = torch.linspace(0, 1, timesteps) ** 2 * 0.8
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        # 安定性のため、ノイズスケジュールについてログ出力
        print(f"拡散ノイズスケジュール: {beta_schedule}, ステップ数: {timesteps}")
        print(f"ノイズレベル: 最小={betas[0].item():.4f}, 最大={betas[-1].item():.4f}")
        
        self.register_buffer("betas", betas)
    
    def add_noise(self, tokens: torch.Tensor, t: int) -> torch.Tensor:
        """
        入力トークンにノイズを加える
        
        Args:
            tokens: 入力トークン [batch_size, seq_len]
            t: タイムステップ
            
        Returns:
            ノイズを加えたトークン
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # t時点でのノイズ確率
        noise_ratio = self.betas[t]
        
        # マスク確率（ステップに応じて徐々に増加）
        mask = torch.bernoulli(torch.full((batch_size, seq_len), noise_ratio, device=device))
        
        # マスクを適用
        noisy_tokens = tokens.clone()
        noisy_tokens[mask.bool()] = self.mask_token_id
        
        return noisy_tokens
        
    def forward(self, tokens: torch.Tensor, t: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        拡散プロセスの実行
        
        Args:
            tokens: 入力トークン [batch_size, seq_len]
            t: タイムステップ（指定がなければランダムに選択）
            
        Returns:
            (ノイズ入りトークン, ラベル)のタプル
        """
        batch_size = tokens.shape[0]
        device = tokens.device
        
        # タイムステップが指定されていない場合はランダムに選択
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        
        # ランダムにノイズを加える（batch内の各要素に対して異なるtを適用）
        noisy_tokens = torch.zeros_like(tokens)
        labels = torch.zeros_like(tokens)
        
        for i in range(batch_size):
            noisy_tokens[i] = self.add_noise(tokens[i].unsqueeze(0), t[i].item()).squeeze(0)
            labels[i] = tokens[i]  # 元のトークンをラベルとする
            
        # ノイズを加えた場所のみラベルにする（他は-100で無視）
        mask = (noisy_tokens == self.mask_token_id)
        final_labels = torch.ones_like(labels) * -100
        final_labels[mask] = labels[mask]
        
        return noisy_tokens, final_labels


