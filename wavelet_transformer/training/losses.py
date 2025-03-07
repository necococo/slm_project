"""
カスタム損失関数の実装
特にCut Cross Entropyなど、効率的な損失計算のための関数を提供
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

def linear_cross_entropy(
    hidden_states: torch.Tensor, 
    weight: torch.Tensor, 
    target: torch.Tensor, 
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    線形時間の交差エントロピー計算（Cut Cross Entropy）
    
    Args:
        hidden_states: モデルの埋め込み出力 [batch_size, seq_len, hidden_size]
        weight: 分類器の重み行列 [vocab_size, hidden_size]
        target: ターゲットラベル [batch_size, seq_len]
        label_smoothing: ラベルスムージングの強さ（0.0-1.0）
        
    Returns:
        損失値
    """
    # 入力チェック: hidden_statesとweightが同じデータ型（float16など）であることを確認
    if hidden_states.dtype != weight.dtype:
        weight = weight.to(dtype=hidden_states.dtype)
    
    # 入力形状の取得
    bsz, seq_len, hidden_size = hidden_states.shape
    target_classes = target.view(-1)
    
    # -100はマスクトークン（無視する位置）
    valid_indices = target_classes != -100
    if not valid_indices.any():
        return hidden_states.new_zeros(1, requires_grad=True)
    
    # 有効な位置だけを抽出
    hidden_states = hidden_states.view(-1, hidden_size)[valid_indices]
    target_classes = target_classes[valid_indices]
    
    # 対象クラスの埋め込みを取得
    target_weight = weight[target_classes]
    
    # 内積計算
    logits_target = torch.sum(hidden_states * target_weight, dim=-1)
    
    # 完全なロジットを計算せずに損失を計算
    # log_sum_expをサンプリングで近似
    sample_size = min(512, weight.size(0))  # 語彙からサンプリングするサイズ
    
    # ランダムにネガティブサンプリング
    noise_indx = torch.randint(
        0, weight.size(0), (hidden_states.size(0), sample_size), device=hidden_states.device
    )
    
    # サンプルした単語埋め込みと内積を計算
    noise_weight = weight[noise_indx]
    logits_noise = torch.bmm(
        noise_weight, hidden_states.unsqueeze(-1)
    ).squeeze(-1)
    
    # ラベルスムージングの適用
    if label_smoothing > 0.0:
        # ターゲットクラスに対する信頼度を調整
        target_prob = 1.0 - label_smoothing
        # サンプリングしたクラスに対する確率を均等配分
        noise_prob = label_smoothing / sample_size
    else:
        target_prob = 1.0
        noise_prob = 0.0
    
    # ロジットを結合（ターゲットを含む）
    # 最初の列がターゲットのロジット、残りが負例のロジット
    all_logits = torch.cat([logits_target.unsqueeze(-1), logits_noise], dim=-1)
    
    # ソフトマックスクロスエントロピーの計算
    log_softmax_all = F.log_softmax(all_logits, dim=-1)
    loss = -target_prob * log_softmax_all[:, 0]  # ターゲットに対する対数確率
    
    # ラベルスムージングがある場合、ノイズサンプルにも確率を割り当てる
    if noise_prob > 0:
        loss = loss - noise_prob * log_softmax_all[:, 1:].sum(dim=-1)
    
    return loss.mean()

class CutCrossEntropyLoss(nn.Module):
    """
    Cut Cross Entropyの損失モジュール実装
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        weight: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        順伝播処理
        
        Args:
            hidden_states: モデルの埋め込み出力 [batch_size, seq_len, hidden_size]
            weight: 分類器の重み行列 [vocab_size, hidden_size]
            target: ターゲットラベル [batch_size, seq_len]
            
        Returns:
            損失値
        """
        return linear_cross_entropy(
            hidden_states=hidden_states,
            weight=weight,
            target=target,
            label_smoothing=self.label_smoothing
        )

class WaveletRegularizationLoss(nn.Module):
    """
    Wavelet係数にスパース正則化を適用する損失
    
    Waveletの本質的な特徴であるスパース性を促進するための損失関数
    """
    def __init__(self, alpha: float = 0.001):
        """
        Args:
            alpha: 正則化の強さ
        """
        super().__init__()
        self.alpha = alpha
        
    def forward(self, wavelet_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Wavelet係数のスパース正則化損失を計算
        
        Args:
            wavelet_coeffs: Wavelet係数 [batch_size, seq_len, hidden_size]
            
        Returns:
            L1正則化損失
        """
        # L1正則化（スパース性を促進）
        return self.alpha * torch.abs(wavelet_coeffs).mean()
