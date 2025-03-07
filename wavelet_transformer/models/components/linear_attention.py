"""
線形計算量O(N)の注意機構
標準的な注意機構のO(N²)計算量を改善するための実装
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List

class LinearAttention(nn.Module):
    """
    線形計算量(O(N))の注意機構
    
    標準的な注意機構: Attention(Q, K, V) = softmax(QK^T)V
    計算量: O(N²)
    
    線形注意機構: φ(Q)φ(K)^TV
    計算量: O(N)
    
    参考文献: Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
    https://arxiv.org/abs/2006.16236
    """
    def __init__(
        self, 
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        feature_map: str = "elu",  # 特徴マップ関数 ("elu", "relu", "favor+")
        eps: float = 1e-6,
    ):
        """
        Args:
            hidden_size: 隠れ層の次元数
            num_heads: 注意ヘッドの数
            dropout: ドロップアウト率
            feature_map: 使用する特徴マップ関数
            eps: 数値安定性のための小さな値
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.eps = eps
        self.feature_map = feature_map
        
        # 通常の線形射影
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # 出力射影
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def _feature_map_elu(self, x: torch.Tensor) -> torch.Tensor:
        """
        ELU+1特徴マップ関数
        φ(x) = elu(x) + 1
        """
        return F.elu(x) + 1
    
    def _feature_map_relu(self, x: torch.Tensor) -> torch.Tensor:
        """
        ReLU特徴マップ関数
        φ(x) = relu(x)
        """
        return F.relu(x)
    
    def _feature_map_favor_plus(self, x: torch.Tensor) -> torch.Tensor:
        """
        FAVOR+特徴マップ（正規乱数による近似ランダム特徴）
        
        参考: Rethinking Attention with Performers
        https://arxiv.org/abs/2009.14794
        """
        # シーケンス長と特徴次元
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        # グローバルに共有されるランダム射影行列
        if not hasattr(self, '_random_proj'):
            # ランダム射影行列の初期化（最初の実行時のみ）
            # サイズ: [head_dim, head_dim]
            rand_matrix = torch.randn(head_dim, head_dim, device=x.device) * 0.02
            self.register_buffer('_random_proj', rand_matrix)
        
        # exp(x)の期待値が1になるよう正規化
        x_norm = x / math.sqrt(self.head_dim)
        
        # ランダム特徴に射影
        projection = torch.matmul(x, self._random_proj)
        
        # 正と負の部分に分けて指数関数を計算
        x_pos = torch.exp(projection)
        x_neg = torch.exp(-projection)
        
        # 結合して返す
        return torch.cat([x_pos, x_neg], dim=-1)
        
    def _apply_feature_map(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        クエリとキーに特徴マップを適用
        
        Args:
            q: クエリテンソル [batch_size, num_heads, seq_len, head_dim]
            k: キーテンソル [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            特徴マップ適用後の(q, k)
        """
        if self.feature_map == "elu":
            return self._feature_map_elu(q), self._feature_map_elu(k)
        elif self.feature_map == "relu":
            return self._feature_map_relu(q), self._feature_map_relu(k)
        elif self.feature_map == "favor+":
            return self._feature_map_favor_plus(q), self._feature_map_favor_plus(k)
        else:
            raise ValueError(f"不明な特徴マップ: {self.feature_map}")
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        線形注意機構のフォワードパス
        
        Args:
            x: 入力テンソル [batch_size, seq_len, hidden_size]
            attention_mask: 注意マスク [batch_size, 1, 1, seq_len] （オプション）
            
        Returns:
            出力テンソル [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # クエリ・キー・バリューの射影
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # トランスポーズ: [batch_size, num_heads, seq_len, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # カーネル特徴マップを適用
        q_mapped, k_mapped = self._apply_feature_map(q, k)
        
        # 注意マスクの適用（必要な場合）
        if attention_mask is not None:
            # マスクを拡張し、キーの特徴マップに適用
            mask = attention_mask.unsqueeze(-1)  # [batch_size, 1, 1, seq_len, 1]
            k_mapped = k_mapped * mask
        
        # 線形注意計算
        # Step 1: K のグローバル集約 (Σ_j φ(k_j))
        kv = torch.einsum('bnsd,bnse->bnde', k_mapped, v)  # [batch, heads, head_dim, value_dim]
        
        # Step 2: 正規化係数 (Σ_j φ(k_j))
        z = torch.einsum('bnsd,bnsd->bns', q_mapped, torch.sum(k_mapped, dim=2))  # [batch, heads, seq_len]
        z = z.unsqueeze(-1)  # [batch, heads, seq_len, 1]
        
        # 数値安定性のため
        z = torch.clamp(z, min=self.eps)
        
        # Step 3: 最終的な注意出力 (φ(q_i)^T Σ_j φ(k_j)v_j)
        attention_output = torch.einsum('bnsd,bnde->bnse', q_mapped, kv)  # [batch, heads, seq_len, value_dim]
        
        # 正規化
        attention_output = attention_output / z
        
        # 形状を元に戻す: [batch_size, seq_len, hidden_size]
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.hidden_size)
        
        # 出力射影
        output = self.output_proj(attention_output)
        output = self.dropout(output)
        
        return output


class FFTAttention(nn.Module):
    """
    高速フーリエ変換(FFT)を使った線形計算量の注意機構
    
    畳み込み定理を利用して注意計算をO(N log N)に効率化
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_size: 隠れ層の次元数
            num_heads: 注意ヘッドの数
            dropout: ドロップアウト率
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # 線形射影
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # 出力射影
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # スケーリング係数
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        FFTベースの注意機構のフォワードパス
        
        Args:
            x: 入力テンソル [batch_size, seq_len, hidden_size]
            attention_mask: 無視（この実装では使用しない）
            
        Returns:
            出力テンソル [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # クエリ・キー・バリューの射影
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # トランスポーズ: [batch_size, num_heads, seq_len, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # FFTを使った線形時間計算
        # 1. 実数FFTを計算
        q_fft = torch.fft.rfft(q, dim=2)
        k_fft = torch.fft.rfft(k, dim=2)
        v_fft = torch.fft.rfft(v, dim=2)
        
        # 2. 周波数領域での乗算（畳み込み→内積に相当）
        output_fft = q_fft * k_fft.conj() * v_fft
        
        # 3. 逆FFTで時間領域に戻す
        output = torch.fft.irfft(output_fft, n=seq_len, dim=2)
        
        # スケーリング
        output = output * self.scale
        
        # 形状を元に戻す: [batch_size, seq_len, hidden_size]
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_size)
        
        # 出力射影
        output = self.output_proj(output)
        output = self.dropout(output)
        
        return output


class WaveletLinearAttention(nn.Module):
    """
    Wavelet変換と線形注意機構を組み合わせた効率的な注意機構
    
    異なる時間スケールの情報を効率的に処理
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        wavelet_levels: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_size: 隠れ層の次元数
            num_heads: 注意ヘッドの数
            wavelet_levels: Wavelet分解レベル
            dropout: ドロップアウト率
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.wavelet_levels = wavelet_levels
        
        # 線形射影
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # マルチスケール処理のための畳み込み層
        self.wavelet_convs = nn.ModuleList([
            nn.Conv1d(
                hidden_size, 
                hidden_size, 
                kernel_size=2**i + 1, 
                padding=2**i // 2
            ) for i in range(1, wavelet_levels + 1)
        ])
        
        # スケール重み
        self.scale_weights = nn.Parameter(torch.ones(wavelet_levels))
        
        # 線形注意機構
        self.linear_attention = LinearAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 出力射影
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Wavelet線形注意機構のフォワードパス
        
        Args:
            x: 入力テンソル [batch_size, seq_len, hidden_size]
            attention_mask: 注意マスク（オプション）
            
        Returns:
            出力テンソル [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # マルチスケール特徴抽出（Wavelet風）
        x_trans = x.transpose(1, 2)  # [batch_size, hidden_size, seq_len]
        
        # 各スケールでの畳み込み
        wavelet_outputs = []
        for conv in self.wavelet_convs:
            wavelet_out = conv(x_trans)
            # サイズ調整
            if wavelet_out.size(2) != seq_len:
                wavelet_out = F.interpolate(wavelet_out, size=seq_len, mode='linear')
            wavelet_outputs.append(wavelet_out)
        
        # スケール間の重み付け
        scale_weights = F.softmax(self.scale_weights, dim=0)
        
        # 重み付き和
        multi_scale_features = torch.zeros_like(x_trans)
        for i, out in enumerate(wavelet_outputs):
            multi_scale_features = multi_scale_features + scale_weights[i] * out
        
        # 元の形状に戻す
        multi_scale_features = multi_scale_features.transpose(1, 2)  # [batch_size, seq_len, hidden_size]
        
        # マルチスケール特徴を使って線形注意を計算
        attention_output = self.linear_attention(multi_scale_features, attention_mask)
        
        # 出力射影
        output = self.output_proj(attention_output)
        output = self.dropout(output)
        
        return output
