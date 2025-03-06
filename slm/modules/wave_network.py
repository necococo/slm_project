"""
Wave Network - An Ultra-Small Language Model (https://arxiv.org/abs/2411.02674v4)
この実装はFigure 6(a)(b)に基づき、波の加算による干渉表現とRoPE位置エンコーディングを使用しています。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from slm.modules.rmsnorm import RMSNorm
from slm.modules.rope import RoPEEmbedding
from slm.modules.activations import GatedMLP  # GatedMLPをインポート
from slm.config import ModelConfig

def compute_wave_representation(x: torch.Tensor, global_mode: bool = False, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    入力テンソルを波表現（実部と虚部）に変換
    
    Args:
        x: 入力テンソル [batch_size, seq_len, dim]
        global_mode: Trueなら文レベル(グローバル)、Falseならトークンレベル(ローカル)
        eps: 数値安定性のための小さな値
        
    Returns:
        (real_part, imag_part): 波表現の実部と虚部
    """
    # 念の為float32に
    # x = x.float()
    B, S, D = x.shape
    
    # グローバル振幅の計算 (モードによって集約次元が異なる)
    if global_mode:
        # 文レベル: dim=(1, 2) で全体のコンテキスト情報を捉える
        G = torch.sqrt(torch.sum(x * x, dim=(1, 2), keepdim=True) + eps)  # [B, 1, 1]
        G = G.expand(-1, S, D)  # [B, S, D]
    else:
        # トークンレベル: dim=1 で各次元の特徴量を保存する
        G = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + eps)  # [B, 1, D]
        G = G.expand(-1, S, -1)  # [B, S, D]
    
    G_safe = torch.clamp(G, min=eps)
    
    # 比率 (w_jk / G_k)
    ratio = x / G_safe
    ratio = torch.clamp(ratio, -0.99, 0.99)  # 数値安定性のため
    
    # 位相角 (α_jk) の計算
    inside = 1.0 - ratio**2
    inside = F.relu(inside) + eps  # 負値を除去
    
    # arctan2(√(1-ratio²), ratio)
    alpha = torch.atan2(torch.sqrt(inside), ratio)
    
    # 波表現への変換
    real_part = G_safe * torch.cos(alpha)
    imag_part = G_safe * torch.sin(alpha)
    
    return real_part, imag_part

class WaveLayer(nn.Module):
    """
    Wave Network Layer の実装 (Fig.6(a))
    """
    def __init__(self, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Feed Forward Network - 効率的なGatedMLPを使用
        self.ffn = GatedMLP(hidden_size * 2)
        
        # Normalization layers
        self.ffn_norm = RMSNorm(hidden_size * 2)
        
        # Projection back to original dimension
        self.proj = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.final_norm = RMSNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wave Layer forward pass
        
        Args:
            x: 入力テンソル [B, S, D]
            
        Returns:
            処理された出力テンソル [B, S, D]
        """       
        B, S, D = x.shape
        eps = 1e-5
        
        # NaN対策
        if torch.isnan(x).any():
            print("Warning: NaNs in input, replacing with zeros")
            # x = torch.nan_to_num(x, nan=0.0)
        
        # 文レベルのwave表現 (グローバルコンテキスト)
        real_sen, imag_sen = compute_wave_representation(x, global_mode=True, eps=eps)
        
        # トークンレベルのwave表現
        real_token, imag_token = compute_wave_representation(x, global_mode=False, eps=eps)
        
        # 波の干渉（加算）- Fig.6(a)の中央部分
        # (a+bi) + (c+di) = (a+c) + (b+d)i
        combined_real = real_sen + real_token
        combined_imag = imag_sen + imag_token
        
        # 結合波表現
        wave_repr = torch.cat([combined_real, combined_imag], dim=-1)  # [B, S, 2D]
        
        # FFN処理
        ffn_out = self.ffn(wave_repr)
        wave_repr = self.ffn_norm(wave_repr + ffn_out)  # 残差接続
        
        # 元の次元に戻す
        output = self.proj(wave_repr)  # [B, S, D]
        output = output + wave_repr[..., :D]  # 残差接続（前半部分のみ）
        
        output = self.dropout(output)
        output = self.final_norm(output)
        
        return output

class WaveNetworkBlock(nn.Module):
    """
    Wave Network Block の実装 (Fig.6(b))
    """
    def __init__(
        self, 
        hidden_size: int, 
        dropout_prob: float = 0.1,
        use_rope: bool = True,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_rope = use_rope
        
        # Wave Layer
        self.wave_layer = WaveLayer(hidden_size, dropout_prob)
        
        # RoPE (オプション)
        if use_rope:
            self.rope = RoPEEmbedding(hidden_size, max_seq_len)
        
        # 残差接続とノーマライゼーション
        self.dropout = nn.Dropout(dropout_prob)
        self.norm = RMSNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wave Network Block forward pass
        
        Args:
            x: 入力テンソル [B, S, D]
            
        Returns:
            処理された出力テンソル [B, S, D]
        """
        # RoPE位置エンコーディングを適用（オプション）
        if self.use_rope:
            # RoPEを適用する前に適切な形状に変換
            B, S, D = x.shape
            x_4d = x.view(B, S, 1, D)  # [B, S, 1, D]
            x_4d_rope = self.rope(x_4d)
            x_rope = x_4d_rope.view(B, S, D)
            wave_input = x_rope
        else:
            wave_input = x
        
        # Wave Layer処理
        wave_output = self.wave_layer(wave_input)
        
        # 残差接続
        output = x + self.dropout(wave_output)
        output = self.norm(output)

        return output

class WaveNetworkLM(nn.Module):
    """
    Wave Network モデル全体の実装
    """
    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()
        self.config = config
        
        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
        num_layers = config.num_layers
        max_seq_len = config.max_seq_len
        dropout_prob = config.dropout_prob
        use_rope = config.use_rope
        
        # トークン埋め込み　nanが多発するので精緻な計算をするためfloat32にしておく
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, dtype=torch.float32)
        
        # Wave Network Blocksのスタック
        self.layers = nn.ModuleList([
            WaveNetworkBlock(
                hidden_size=hidden_size,
                dropout_prob=dropout_prob,
                use_rope=use_rope,
                max_seq_len=max_seq_len
            ) for _ in range(num_layers)
        ])
        
        # 分類器（cut-cross-entropy用の重み）
        self.classifier = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # 最終ノーマライゼーション
        self.norm = RMSNorm(hidden_size)
        # 初期化 直下の_init_weightsクラス関数をつかう
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        モデルのforward pass
        
        Args:
            input_ids: 入力トークンID [B, S]
            
        Returns:
            hidden_states: 最終的な隠れ状態 [B, S, D]
        """
        # トークン埋め込み
        hidden_states = self.token_embedding(input_ids)

        # 各レイヤーを通過
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        # 最終ノーマライゼーション
        hidden_states = self.norm(hidden_states)

        logits = self.classifier(hidden_states)

        return hidden_states # Cut Cross Entropyでは logitではなく embeddings=hidden_states を入力する

    def get_classifier_weights(self) -> torch.Tensor:
        """
        How:
            cut-cross-entropyで linear_cross_entropy() を呼ぶ際に必要となる
            分類器の重み (V, D) を返す。
        """
        # classifier.weight shape: (V, D)
        return self.classifier.weight