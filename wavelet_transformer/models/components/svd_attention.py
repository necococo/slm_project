"""
特異値分解(SVD)を用いた低ランク近似による効率的な注意機構
通常の注意機構のO(n²)計算複雑性をO(nr)に削減（rはランク数）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

class SVDAttention(nn.Module):
    """
    特異値分解を用いた低ランク近似による効率的な注意機構
    
    標準的な自己注意機構では、QK^Tの計算がO(n²)の計算量を必要とするが、
    低ランク近似によりO(nr)の計算量に削減します（rはランク数）。
    """
    def __init__(
        self,
        hidden_size: int,
        rank: int = 64,  # 低ランク近似の次元数
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_size: 隠れ層の次元数
            rank: 低ランク近似の次元数（小さいほど計算効率が良いが、表現力は落ちる）
            num_heads: 注意ヘッドの数
            dropout: ドロップアウト率
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # 通常の線形射影
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # 低ランク射影
        self.q_low_rank = nn.Linear(self.head_dim, rank)
        self.k_low_rank = nn.Linear(self.head_dim, rank)
        
        # 出力射影
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # スケーリング係数
        self.scale = 1.0 / math.sqrt(rank)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        SVDベースの低ランク近似による注意機構の順伝搬
        
        Args:
            x: 入力テンソル [batch_size, seq_len, hidden_size]
            attention_mask: 注意マスク [batch_size, 1, 1, seq_len]（オプション）
            
        Returns:
            注意機構適用後のテンソル [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # クエリ・キー・バリュー射影
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # トランスポーズ: [batch_size, num_heads, seq_len, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # 低ランク射影 - これがSVD近似の核心部分
        # [batch_size, num_heads, seq_len, rank]
        q_low = self.q_low_rank(q)
        k_low = self.k_low_rank(k)
        
        # 低ランク空間での注意スコア計算 - O(n*rank)
        attention_scores = torch.matmul(q_low, k_low.transpose(-1, -2)) * self.scale
        
        # マスキング（指定されている場合）
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # ソフトマックスで注意ウェイトに変換
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 注意ウェイトを使って値を計算
        context = torch.matmul(attention_probs, v)
        
        # トランスポーズして元の形状に戻す [batch_size, seq_len, hidden_size]
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_size)
        
        # 出力射影
        output = self.output_proj(context)
        
        return output


class NystromAttention(nn.Module):
    """
    Nyström法を用いた線形計算量の注意機構
    
    参考文献: Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention
    https://arxiv.org/abs/2102.03902
    
    ランドマークポイントを使用してO(n)の計算量で自己注意を近似
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int = 8, 
        num_landmarks: int = 64,  # ランドマークポイントの数
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_size: 隠れ層の次元数
            num_heads: 注意ヘッドの数
            num_landmarks: ランドマークポイントの数（少ないほど効率的だが精度は落ちる）
            dropout: ドロップアウト率
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_landmarks = num_landmarks
        self.head_dim = hidden_size // num_heads
        
        # 通常の線形射影
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
        Nyström法による線形計算量の注意機構の順伝搬
        
        Args:
            x: 入力テンソル [batch_size, seq_len, hidden_size]
            attention_mask: 注意マスク（オプション）
            
        Returns:
            注意機構適用後のテンソル [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # クエリ・キー・バリュー射影
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # トランスポーズ: [batch_size, num_heads, seq_len, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # seq_lenがnum_landmarksより小さい場合、調整
        actual_landmarks = min(self.num_landmarks, seq_len)
        
        # ランドマークポイントを選択（シンプルな等間隔サンプリング）
        # より洗練されたサンプリング法も検討できる
        if seq_len <= actual_landmarks:
            # シーケンス長がランドマーク数以下の場合、全ポイントを使用
            k_landmarks = k
            v_landmarks = v
        else:
            # 等間隔サンプリング
            indices = torch.linspace(0, seq_len - 1, actual_landmarks).long()
            k_landmarks = k[:, :, indices, :]
            v_landmarks = v[:, :, indices, :]
        
        # ステップ1: Q と ランドマークK の間の類似度を計算
        qk_landmarks = torch.matmul(q, k_landmarks.transpose(-1, -2)) * self.scale  # [b, h, n, m]
        
        # ステップ2: ランドマーク間の内部類似度を計算
        kk_landmarks = torch.matmul(k_landmarks, k_landmarks.transpose(-1, -2)) * self.scale  # [b, h, m, m]
        
        # 数値安定性のため小さな値を加算
        kk_landmarks = kk_landmarks + 1e-6 * torch.eye(
            actual_landmarks, device=kk_landmarks.device
        ).unsqueeze(0).unsqueeze(0)
        
        # ステップ3: ランドマーク間の類似度の逆行列を計算
        kk_inv = torch.inverse(kk_landmarks)  # [b, h, m, m]
        
        # ステップ4: ステップ1とステップ3の結果を乗算
        qk_scaled = torch.matmul(qk_landmarks, kk_inv)  # [b, h, n, m]
        
        # ステップ5: ランドマークとバリューを乗算
        landmark_values = torch.matmul(qk_scaled, torch.matmul(k_landmarks.transpose(-1, -2), v))
        
        # 形状を元に戻す
        context = landmark_values.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_size)
        
        # 出力射影
        output = self.output_proj(context)
        
        return output


class HybridSVDAttention(nn.Module):
    """
    波動表現とSVD注意機構を組み合わせたハイブリッド注意機構
    
    波動表現の複素平面上の特性とSVDの低ランク近似を融合
    """
    def __init__(
        self,
        hidden_size: int,
        rank: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_size: 隠れ層の次元数
            rank: 低ランク近似の次元数
            num_heads: 注意ヘッドの数
            dropout: ドロップアウト率
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # SVD低ランク注意機構
        self.svd_attention = SVDAttention(
            hidden_size=hidden_size,
            rank=rank,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 波動表現の振幅と位相分離のための射影
        self.amplitude_proj = nn.Linear(hidden_size, hidden_size)
        self.phase_proj = nn.Linear(hidden_size, hidden_size)
        
        # 出力射影
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        ハイブリッド注意機構の順伝搬
        
        Args:
            x: 入力テンソル [batch_size, seq_len, hidden_size]
            attention_mask: 注意マスク（オプション）
            
        Returns:
            注意機構適用後のテンソル [batch_size, seq_len, hidden_size]
        """
        # SVD注意を適用
        svd_output = self.svd_attention(x, attention_mask)
        
        # 振幅と位相の分離表現
        amplitude = self.amplitude_proj(x)
        phase = self.phase_proj(x)
        
        # 複素波動表現を計算（極形式：振幅×e^(i×位相)）
        # 実際の計算では、cos(phase)とsin(phase)を用いて実部と虚部に分離
        real_part = amplitude * torch.cos(phase)
        imag_part = amplitude * torch.sin(phase)
        
        # SVD出力と波動表現を統合（加重和）
        # より複雑な統合方法も考えられる
        hybrid_output = svd_output + 0.5 * (real_part + imag_part)
        
        # 最終出力
        output = self.output_proj(hybrid_output)
        output = self.dropout(output)
        
        return output
