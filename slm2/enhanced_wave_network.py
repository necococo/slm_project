"""
トークン間の関係を直接波表現でモデル化する拡張Wave Network
標準的なWave Networkを拡張した実験的実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from slm.config import ModelConfig
from slm.modules.wave_network import WaveNetworkLM, compute_wave_representation
from slm2.wave_attention import WaveInteractionAttention, WaveletAttention, PhaseInteractionLayer, WaveNetworkEnhanced
from slm.modules.rmsnorm import RMSNorm

class EnhancedWaveNetworkLM(WaveNetworkLM):
    """
    波動に基づくトークン間関係のモデリングを強化したWave Network
    """
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        hidden_size = config.hidden_size
        num_layers = config.num_layers
        dropout_prob = config.dropout_prob
        
        # 標準のレイヤーを拡張レイヤーで置き換え
        self.enhanced_layers = nn.ModuleList([
            WaveNetworkEnhanced(hidden_size, dropout_prob)
            for _ in range(num_layers)
        ])
        
        # 波動混合重み（学習可能）
        self.global_mix_param = nn.Parameter(torch.tensor(0.5))
        self.token_mix_param = nn.Parameter(torch.tensor(0.5))
        
    def enhanced_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        拡張フォワードパス - 波動関係モデリングを強化
        
        Args:
            input_ids: 入力トークンID [B, S]
            
        Returns:
            hidden_states: 最終的な隠れ状態 [B, S, D]
        """
        # トークン埋め込み
        hidden_states = self.token_embedding(input_ids)
        
        # RoPEを最初に一度だけ適用
        if self.use_rope:
            B, S, D = hidden_states.shape
            x_4d = hidden_states.view(B, S, 1, D)
            x_4d_rope = self.rope(x_4d)
            hidden_states = x_4d_rope.view(B, S, D)

        # 拡張レイヤーを通過
        for layer in self.enhanced_layers:
            hidden_states = layer(hidden_states)
            
        # 最終ノーマライゼーション
        hidden_states = self.norm(hidden_states)

        return hidden_states
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        オーバーライドしたforward - 拡張版を使用
        
        Args:
            input_ids: 入力トークンID [B, S]
            
        Returns:
            hidden_states: 最終的な隠れ状態 [B, S, D]
        """
        return self.enhanced_forward(input_ids)
