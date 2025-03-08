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
    # 数値安定性のためにepsを調整
    eps = 1e-4  # 1e-3などより大きな値も試してみる価値があります
    
    # 念の為float32に強制変換し、NaNをチェック
    x = x.float()
    if torch.isnan(x).any():
        x = torch.nan_to_num(x, nan=0.0)
    
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
    
    # 比率計算の数値安定性強化 - タンジェントのスケールを調整
    ratio = x / G_safe
    ratio = torch.tanh(ratio) * 0.95  # 0.99から0.95に変更してより安全なマージンを確保
    
    # 位相角 (α_jk) の計算
    inside = 1.0 - ratio**2
    inside = F.relu(inside) + eps  # 負値を除去
    
    # arctan2(√(1-ratio²), ratio)
    alpha = torch.atan2(torch.sqrt(inside), ratio)
    
    # 波表現への変換
    real_part = G_safe * torch.cos(alpha)
    imag_part = G_safe * torch.sin(alpha)
    
    # 最終的な数値チェック
    real_part = torch.nan_to_num(real_part)
    imag_part = torch.nan_to_num(imag_part)
    
    return real_part, imag_part

class SingleWaveLayer(nn.Module):
    """
    Wave Network Layer の実装 (Fig.6(a))
    論文の図6(a)に忠実に従った実装
    """
    def __init__(self, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 拡張係数を大きくして表現力を向上
        expansion_factor = 6  # 4から6に増加
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2 * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout_prob),  # ドロップアウトを追加
            nn.Linear(hidden_size * 2 * expansion_factor, hidden_size * 2)
        )
        
        # 論文の図6(a)では明示的なノーマライゼーションは表示されていない
        # ただし、波表現から元の埋め込み空間への変換が示されている
        self.to_embedding = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wave Layer forward pass
        
        Args:
            x: 入力テンソル [B, S, D]
            
        Returns:
            output: 処理後のテンソル [B, S, D]
        """       
        B, S, D = x.shape
        eps = 1e-5
        
        # NaN対策
        if torch.isnan(x).any():
            print("Warning: NaNs in input, replacing with zeros")
        
        # 文レベルのwave表現 (グローバルコンテキスト)
        real_sen, imag_sen = compute_wave_representation(x, global_mode=True, eps=eps)
        
        # トークンレベルのwave表現
        real_token, imag_token = compute_wave_representation(x, global_mode=False, eps=eps)
        
        # 波の干渉（加算）- Fig.6(a)の中央部分
        combined_real = real_sen + real_token
        combined_imag = imag_sen + imag_token
        
        # 結合波表現
        wave_repr = torch.cat([combined_real, combined_imag], dim=-1)  # [B, S, 2D]
        
        # FFN処理 - 論文の図6(a)に従い波表現空間で処理
        wave_repr = self.ffn(wave_repr)  # [B, S, 2D]
        
        # 波表現から埋め込み空間への変換（図6(a)最後のステップ）
        output = self.to_embedding(wave_repr)  # [B, S, D]
        output = self.norm(output)

        return output

class WaveNetworkBlock(nn.Module):
    """
    Wave Network Block の実装 (Fig.6(b))
    論文の図6(b)に忠実に従った実装
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
        
        # Wave Layer - 波表現処理のみ
        self.wave_layer = SingleWaveLayer(hidden_size, dropout_prob)
        
        # RoPE (オプション)
        if self.use_rope:
            self.rope = RoPEEmbedding(hidden_size, max_seq_len)
                
        # 残差接続後の処理
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wave Network Block forward pass
        
        Args:
            x: 入力テンソル [B, S, D]
            
        Returns:
            処理された出力テンソル [B, S, D]
        """
        # RoPE位置エンコーディングを適用（各層で適用）
        if self.use_rope:
            B, S, D = x.shape
            x_4d = x.view(B, S, 1, D)  # [B, S, 1, D] - RoPE用に形状変更
            x_4d_rope = self.rope(x_4d)  # RoPE適用 - 次元保存 [B, S, 1, D]
            wave_input = x_4d_rope.view(B, S, D)  # 元の形状に戻す [B, S, D]
        else:
            wave_input = x
        
        # 1. Wave Layer処理
        wave_output = self.wave_layer(wave_input)  # [B, S, D]
        
        # 2. 残差接続とPost-Norm（論文の図6(b)に従う）
        output = self.dropout(x + wave_output)
        
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
        # cut cross entropyを使用するかどうかのフラグ（デフォルトはTrue）
        self.use_cut_cross_entropy = getattr(config, 'use_cut_cross_entropy', True)
        
        # トークン埋め込み　nanが多発するので精緻な計算をするためfloat32にしておく
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, dtype=torch.float32)
        
        # Wave Network Blocksのスタック - 各レイヤーでRoPEを使用する設定
        self.layers = nn.ModuleList([
            WaveNetworkBlock(
                hidden_size=hidden_size,
                dropout_prob=dropout_prob,
                use_rope=use_rope,  # configから取得した設定値を使用
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
            use_cut_cross_entropy=Trueの場合: hidden_states [B, S, D]
            use_cut_cross_entropy=Falseの場合: logits [B, S, V]
        """
        # トークン埋め込み
        hidden_states = self.token_embedding(input_ids)
        
        # 各レイヤーを通過（各レイヤーでRoPEを適用）
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        # 最終ノーマライゼーション
        last_hidden_states = self.norm(hidden_states)

        # 損失関数タイプに基づいて異なる出力を返す
        if self.use_cut_cross_entropy:
            # Cut Cross Entropyの場合はlast_hidden_statesを返す
            return last_hidden_states
        else:
            # 通常のCross Entropyの場合はlogitsを返す
            logits = self.classifier(last_hidden_states)
            return logits

    def get_classifier_weights(self) -> torch.Tensor:
        """
        How:
            cut-cross-entropyで linear_cross_entropy() を呼ぶ際に必要となる
            分類器の重み (V, D) を返す。
        """
        # classifier.weight shape: (V, D)
        return self.classifier.weight